"""Bind a learned coverage router to one immutable compact basis bank.

The learned scores only order candidates.  This module pins the exact bank
manifest, candidate order, GEM fingerprints and LP feature layout that were
used to train those scores before uploading the small inference arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .selected_lp_features import FIELDS

ORIGINAL_ROUTER_STAGES=('maxmin','aggregate','exchange')


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be a lowercase SHA256')
    return value


def _host(value):
    return np.asarray(value.get() if hasattr(value, 'get') else value)


def compact_candidate_ids(stage_manifest):
    """Return ordered, collision-resistant IDs for every compact candidate."""
    if not isinstance(stage_manifest, dict):
        raise ValueError('A compact stage manifest is required')
    entries = stage_manifest.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('The compact stage must contain candidate entries')
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Malformed compact candidate entry')
        filename, digest = entry.get('filename'), entry.get('sha256')
        if (not isinstance(filename, str) or not filename
                or Path(filename).name != filename):
            raise ValueError('Candidate filename must be a local basename')
        _sha256(digest, 'candidate sha256')
        result.append(filename + ':' + digest)
    if len(set(result)) != len(result):
        raise ValueError('Compact candidate identities must be unique')
    return tuple(result)


def compact_feature_schema(bank):
    """Return the canonical selected-LP feature schema and its SHA256."""
    try:
        rows, columns = map(int, bank.host_a.shape)
        variable_rows = _host(bank.variable_rows)
        indices = _host(bank.feature_indices)
    except Exception as error:
        raise ValueError('Malformed compact bank feature metadata') from error
    if min(rows, columns) < 1:
        raise ValueError('Compact LP shape must be positive')
    if (variable_rows.ndim != 1 or variable_rows.dtype.kind not in 'iu'
            or np.any(variable_rows < 0) or np.any(variable_rows >= rows)):
        raise ValueError('Invalid compact variable-row metadata')
    shapes = dict(rhs=(rows,), lower=(columns,), upper=(columns,), c=(columns,),
                  delta=(len(variable_rows), columns), col_scale=(columns,),
                  row_scale=(rows,))
    width = sum(int(np.prod(shapes[name], dtype=np.int64)) for name in FIELDS)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or np.any(indices < 0) or np.any(indices >= width)):
        raise ValueError('Invalid compact feature indices')
    schema = dict(
        version=1,
        field_order=list(FIELDS),
        field_shapes={name:list(shapes[name]) for name in FIELDS},
        original_feature_width=width,
        selected_feature_indices=[int(value) for value in indices],
        selected_feature_width=len(indices),
        encoding='nan_to_num(0,+1e13,-1e13);sign(x)*log1p(abs(x));float32',
        flattening='C-order within fields, then field_order concatenation',
    )
    encoded = json.dumps(schema, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return schema, hashlib.sha256(encoded).hexdigest()


def compact_router_provenance(bank_manifest_sha256, stage_manifest, bank,
                              coverage_sha256, model_fingerprints):
    """Build the exact provenance used by both training and deployment."""
    _sha256(bank_manifest_sha256, 'bank_manifest_sha256')
    _sha256(coverage_sha256, 'coverage_sha256')
    if (not isinstance(model_fingerprints, dict) or not model_fingerprints
            or any(not isinstance(name, str) or not name for name in model_fingerprints)):
        raise ValueError('Named model fingerprints are required')
    models = {}
    for name, digest in model_fingerprints.items():
        models[name] = _sha256(digest, f'model_fingerprints.{name}')
    stage = stage_manifest.get('stage') if isinstance(stage_manifest, dict) else None
    key = stage_manifest.get('key') if isinstance(stage_manifest, dict) else None
    if stage not in ORIGINAL_ROUTER_STAGES or not isinstance(key, (list, tuple)) or len(key)!=4:
        raise ValueError('An original compact stage and complete stage key are required')
    key = list(key)
    if (key[0]!=stage or any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
                            for value in key[1:])
            or tuple(key[1:3])!=tuple(bank.host_a.shape) or not 0<=key[3]<=key[1]
            or getattr(bank,'neq',key[3])!=key[3]):
        raise ValueError('Compact stage key mismatch with the actual LP layout')
    key = [stage,*map(int,key[1:])]
    _, schema_digest = compact_feature_schema(bank)
    return dict(
        bank_sha256=bank_manifest_sha256,
        coverage_sha256=coverage_sha256,
        feature_schema_sha256=schema_digest,
        model_fingerprints=models,
        candidate_ids=list(compact_candidate_ids(stage_manifest)),
        stage=stage,
        stage_key=key,
    )


def load_bound_coverage_router(path, expected_sha256, *, bank_manifest_sha256,
                               stage_manifest, bank, model_fingerprints, xp):
    """Load, validate and upload one router; return it and report metadata.

    The training-coverage digest is contained in the externally SHA-pinned
    router.  All deployment-state identities are independently recomputed.
    """
    from .coverage_router import CoverageRouter

    path = Path(path)
    _sha256(expected_sha256, 'expected_sha256')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Coverage router artifact SHA256 mismatch')
    # Read only JSON metadata from a non-pickle artifact to obtain the pinned
    # training-coverage digest needed by CoverageRouter's full provenance API.
    with np.load(path, allow_pickle=False) as data:
        if ('metadata' not in data or data['metadata'].shape != ()
                or data['metadata'].dtype.kind != 'U'):
            raise ValueError('Coverage router requires scalar JSON metadata')
        try:
            metadata = json.loads(str(data['metadata']))
        except (TypeError, ValueError) as error:
            raise ValueError('Malformed coverage-router metadata') from error
    declared = metadata.get('provenance') if isinstance(metadata, dict) else None
    if not isinstance(declared, dict):
        raise ValueError('Coverage router lacks provenance')
    coverage_sha256 = declared.get('coverage_sha256')
    expected = compact_router_provenance(bank_manifest_sha256, stage_manifest,
        bank, coverage_sha256, model_fingerprints)
    router = CoverageRouter.load(path, expected_sha256=expected_sha256,
                                 expected_provenance=expected)
    # CoverageRouter's stable core checks five fields.  Also bind the explicit
    # stage/key extensions created by our shared training helper.
    for name, value in expected.items():
        if router.provenance.get(name) != value:
            raise ValueError(f'Coverage-router binding mismatch: {name}')
    if (router.candidate_count != len(bank.evaluators)
            or router.input_dim != int(_host(bank.feature_indices).size)):
        raise ValueError('Coverage-router dimensions disagree with compact bank')
    device = router.to_device(xp)
    schema, schema_digest = compact_feature_schema(bank)
    record = dict(
        path=str(path.resolve()), sha256=expected_sha256,
        stage=expected['stage'], stage_key=expected['stage_key'],
        bank_manifest_sha256=bank_manifest_sha256,
        coverage_sha256=coverage_sha256,
        feature_schema_sha256=schema_digest,
        feature_schema=schema,
        candidate_count=router.candidate_count,
        candidate_ids=expected['candidate_ids'],
        architecture=router.metadata['architecture'],
        scope='Candidate ordering only; every returned LP still requires the original certificate.',
    )
    return device, record
