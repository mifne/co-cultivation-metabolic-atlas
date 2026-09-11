"""Validated training-only normalized LP inputs for compact-bank rebuilding.

The legacy cache did not hash each NPZ; this loader records current hashes and
checks its explicit ordering/schema. It cannot retroactively prove a historical
file was never modified. New rebuilt artifacts pin all source bytes.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix,diags

from .basis_coverage_selection import validate_training_scope
from .gpu_certified_basis import NormalizedLP
from .selected_lp_features import SelectedLPFeatures

FIELDS=('rhs','lower','upper','c','delta','col_scale','row_scale')
COMPACT_TRAINING_STAGES=('maxmin','aggregate','exchange')


def validate_compact_stage(stage):
    if not isinstance(stage,str) or stage not in COMPACT_TRAINING_STAGES:
        raise ValueError('Stage must be maxmin, aggregate or exchange')
    return stage


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def checked_child(directory,filename,digest=None):
    directory=Path(directory).resolve()
    path=(directory/filename).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Artifact path is outside source directory or missing')
    if digest is not None and sha256_file(path)!=digest:
        raise ValueError('Artifact SHA-256 mismatch')
    return path


def checked_npz(directory,filename,digest=None):
    with np.load(checked_child(directory,filename,digest),allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


@dataclass
class CompactTrainingSource:
    root: dict
    variable_rows: np.ndarray
    data: dict
    feature_indices: np.ndarray
    feature_scale: np.ndarray
    seeds: tuple
    steps: int
    bank_manifest: dict
    stage_manifest: dict
    provenance: dict

    @property
    def size(self):return len(self.data['rhs'])

    @property
    def stage(self):return validate_compact_stage(self.stage_manifest.get('stage'))

    def batches(self,batch_size=32,indices=None):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError('Positive integer batch_size required')
        indices=np.arange(self.size) if indices is None else np.asarray(indices)
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=self.size):
            raise ValueError('Invalid training row indices')
        for start in range(0,len(indices),batch_size):
            yield {key:value[indices[start:start+batch_size]] for key,value in self.data.items()}

    def normalized_problem(self,index):
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<self.size:
            raise ValueError('Training row out of range')
        a=self.root['a'].tolil(copy=True)
        a[self.variable_rows]=self.root['a'][self.variable_rows]+csr_matrix(self.data['delta'][index])
        a=a.tocsr();a.eliminate_zeros();a.sort_indices()
        return NormalizedLP(a,*(self.data[key][index] for key in ('rhs','lower','upper','c')),
            self.root['neq'],self.data['col_scale'][index],self.data['row_scale'][index])

    def original_problem(self,index):
        q=self.normalized_problem(index)
        a=(diags(1./q.row_scale)@q.a@diags(q.col_scale)).tocsr()
        a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
        return a,q.rhs/q.row_scale,q.lower/q.col_scale,q.upper/q.col_scale,q.c*q.col_scale,q.neq

    def selected_features(self):
        m,n=self.root['a'].shape
        selector=SelectedLPFeatures(self.feature_indices,dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),
            delta=(len(self.variable_rows),n),col_scale=(n,),row_scale=(m,)))
        return np.stack([selector.cpu_features({key:value[i] for key,value in self.data.items()})
                         for i in range(self.size)])


def load_compact_training_source(directory,bank_directory,*,stage='maxmin',forbidden_seeds=(),
                                 expected_model_fingerprints=None):
    stage=validate_compact_stage(stage)
    directory,bank_directory=Path(directory).resolve(),Path(bank_directory).resolve()
    training_path=checked_child(directory,'manifest.json')
    bank_path=checked_child(bank_directory,'manifest.json')
    training=json.loads(training_path.read_text())
    manifest=json.loads(bank_path.read_text())
    if manifest.get('status')!='completed':raise ValueError('Base bank is incomplete')
    fingerprints=manifest['model_fingerprints']
    if expected_model_fingerprints is not None and fingerprints!=expected_model_fingerprints:
        raise ValueError('Current GEM fingerprints differ from the base bank')
    scope=validate_training_scope(training,fingerprints,forbidden_seeds=forbidden_seeds)
    if manifest.get('source_manifest')!=training:
        raise ValueError('Training manifest differs from the base bank source')
    if set(manifest['train_seeds']) & set(forbidden_seeds):
        raise ValueError('Base-bank training seeds overlap forbidden evaluation seeds')
    stages=[row for row in manifest['stages'] if row['stage']==stage]
    if len(stages)!=1:raise ValueError('Exactly one selected stage is required')
    row=stages[0];folder=bank_directory/stage
    raw=checked_npz(folder,'root.npz',row['root_sha256'])
    a=csr_matrix((raw['a_data'],raw['a_indices'],raw['a_indptr']),shape=tuple(raw['a_shape']))
    if not np.isfinite(a.data).all():raise ValueError('Nonfinite root matrix')
    a.check_format(full_check=True)
    if not a.has_canonical_format:raise ValueError('Root CSR must be canonical')
    neq=int(raw['neq']);m,n=a.shape
    if row['key']!=[stage,m,n,neq] or not 0<=neq<=m:raise ValueError('Stage key/root mismatch')
    variables=np.asarray(raw['variable_rows'])
    if (variables.ndim!=1 or variables.dtype.kind not in 'iu' or len(set(variables.tolist()))!=len(variables)
            or np.any(variables<neq) or np.any(variables>=m)):
        raise ValueError('Invalid stage-specific variable rows')
    router=checked_npz(folder,'router.npz',row['router_sha256'])
    shapes=dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),delta=(len(variables),n),col_scale=(n,),row_scale=(m,))
    selector=SelectedLPFeatures(router['indices'],shapes)
    width=selector.selected_width
    if (width<1 or router['scale'].shape!=(width,) or not np.isfinite(router['scale']).all()
            or np.any(router['scale']<=0) or router['centers'].shape!=(len(row['entries']),width)
            or not np.isfinite(router['centers']).all()):
        raise ValueError('Invalid immutable router schema or scaling')
    collector_record=None
    if 'source_sha256' in training:
        collector=checked_child(directory,'sources/train_neural_basis_proposals.py',training['source_sha256'])
        collector_record=dict(filename=str(collector.relative_to(directory)),sha256=sha256_file(collector))
    files=sorted((directory/'inputs'/stage).glob('*.npz'))
    if not files:raise ValueError('Training input chunks missing')
    offset=0;chunks=[];chunk_records=[]
    expected_shapes=dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),delta=(len(variables),n),col_scale=(n,),row_scale=(m,))
    for file in files:
        file=checked_child(directory,str(file.relative_to(directory)))
        if file.stem!=f'{offset:06d}':raise ValueError('Training chunk sequence has a gap or reordered rows')
        chunk=checked_npz(file.parent,file.name)
        if set(chunk)!=set(FIELDS):raise ValueError('Unexpected cached training fields')
        if chunk['rhs'].ndim!=2:raise ValueError('Cached RHS must be a batch matrix')
        count=chunk['rhs'].shape[0]
        if count<1:raise ValueError('Empty training chunk')
        for key,shape in expected_shapes.items():
            value=chunk[key]
            if value.shape!=(count,*shape) or value.dtype!=np.dtype('float64'):
                raise ValueError('Cached training dimensions or dtype mismatch: '+key)
            if key in ('lower','upper'):
                if np.isnan(value).any():raise ValueError('NaN bounds')
            elif not np.isfinite(value).all():raise ValueError('Nonfinite training field: '+key)
            if key.endswith('_scale') and np.any(value<=0):raise ValueError('Nonpositive scale')
        if (np.any(chunk['lower']>chunk['upper']) or np.isposinf(chunk['lower']).any()
                or np.isneginf(chunk['upper']).any()):raise ValueError('Invalid training bounds')
        chunks.append(chunk)
        chunk_records.append(dict(filename=str(file.relative_to(directory)),sha256=sha256_file(file),start=offset,rows=count))
        offset+=count
    if offset!=scope.expected_rows:raise ValueError('Training row count does not match seed/horizon contract')
    data={key:np.concatenate([chunk[key] for chunk in chunks]).astype(np.float64,copy=False) for key in FIELDS}
    provenance=dict(role='training_only_dictionary_reconstruction',stage=stage,stage_key=list(row['key']),
        training_manifest_sha256=sha256_file(training_path),
        base_bank_manifest_sha256=sha256_file(bank_path),model_fingerprints=fingerprints,
        training_seeds=list(scope.seeds),steps=scope.steps,row_order='trajectory_major_then_step',
        forbidden_seeds=sorted(set(forbidden_seeds)),input_chunks=chunk_records,
        collector_source=collector_record,
        row_identity=[dict(index=t*scope.steps+s,seed=seed,step=s+1)
            for t,seed in enumerate(scope.seeds) for s in range(scope.steps)],
        historical_integrity_limit='Legacy cache has no chunk hashes or embedded row IDs/actions. Row identity is inferred from the pinned collector. Fixed-row differences <=2e-12 were discarded by that collector. The complete historical environment/LP generator source was not snapshotted. This is the reconstructed cached LP, not a bit-exact historical raw LP. Four uniform-action trajectories do not establish PPO-distribution or boundary coverage.')
    return CompactTrainingSource(dict(a=a,neq=neq),variables.astype(np.int64),data,
        router['indices'],router['scale'],scope.seeds,scope.steps,manifest,row,provenance)
