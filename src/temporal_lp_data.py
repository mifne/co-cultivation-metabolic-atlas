"""Train-only LP compression for learned warm starts, never a certificate.

All matrix entries and original bounds still go to the full-space corrector.
Feature selection/decoder are fit only on the declared training trajectories.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .lp_trace import load_trace_lp


def signed_log(values):
    values = np.nan_to_num(values, nan=0., posinf=1e13, neginf=-1e13)
    return np.sign(values) * np.log1p(np.abs(values))


def load_trajectories(directory, stage, *, required_role=None):
    directory = Path(directory)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('Completed trace required')
    if required_role is not None and manifest.get('role') != required_role:
        raise ValueError('Trace purpose does not permit this use')
    seeds = manifest['seeds']
    steps = manifest['completed_steps']
    if len(set(seeds)) != len(seeds) or len(set(steps)) != 1 or steps[0] < 1:
        raise ValueError('Unique seeds and equal positive trajectory lengths required')
    if stage not in ('maxmin', 'aggregate', 'exchange'):
        raise ValueError('Unknown LP stage')
    entries = {(e['environment_id'], e['step']): e for e in manifest['entries'] if e['stage'] == stage}
    if len(entries) != len(seeds) * steps[0] or len(entries) != sum(e['stage'] == stage for e in manifest['entries']):
        raise ValueError('Incomplete or duplicated stage trace')
    trajectories = [[load_trace_lp(directory, entries[(env, step)])
                     for step in range(1, steps[0]+1)] for env in range(len(seeds))]
    return manifest, trajectories, hashlib.sha256(path.read_bytes()).hexdigest()


def input_vector(problem, matrix_keys):
    a, rhs, lower, upper, c, _ = problem
    row, col = matrix_keys // a.shape[1], matrix_keys % a.shape[1]
    changing = np.asarray(a[row, col]).ravel() if len(matrix_keys) else np.empty(0)
    return signed_log(np.concatenate((rhs, lower, upper, c, changing)))


def randomized_components(centered, rank, seed):
    """Deterministic randomized SVD on TRAIN data, in float64."""
    rank = min(rank, min(centered.shape))
    width = min(rank+12, min(centered.shape))
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(centered @ rng.normal(size=(centered.shape[1], width)))
    for _ in range(2):
        z, _ = np.linalg.qr(centered.T @ q)
        q, _ = np.linalg.qr(centered @ z)
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    return vt[:rank].copy()


class TemporalLPCodec:
    def __init__(self, arrays, metadata):
        self.arrays, self.metadata = arrays, metadata
        a = arrays
        n, m, _ = metadata['dimensions']
        d = len(a['feature_indices'])
        if (a['components'].ndim != 2 or a['components'].shape[1] != n+m
                or a['target_mean'].shape != (n+m,) or a['target_scale'].shape != (n+m,)
                or a['feature_mean'].shape != (d,) or a['feature_scale'].shape != (d,)
                or a['feature_indices'].dtype.kind not in 'iu' or a['matrix_keys'].dtype.kind not in 'iu'
                or np.any(a['feature_indices'] < 0) or np.any(a['matrix_keys'] < 0)
                or np.any(a['matrix_keys'] >= n*m)
                or np.any(a['feature_indices'] >= m+3*n+len(a['matrix_keys']))
                or np.any(a['feature_scale'] <= 0) or np.any(a['target_scale'] <= 0)
                or ('latent_scale' in a and (a['latent_scale'].shape != (a['components'].shape[0],)
                    or np.any(a['latent_scale'] <= 0)))
                or any(not np.isfinite(value).all() for value in a.values())):
            raise ValueError('Invalid temporal codec arrays')

    @property
    def rank(self):
        return self.arrays['components'].shape[0]

    @property
    def feature_dim(self):
        return 2*len(self.arrays['feature_indices'])+1

    def base_features(self, problem):
        n, m, neq = self.metadata['dimensions']
        if problem[0].shape != (m, n) or problem[-1] != neq:
            raise ValueError('LP shape/phase does not match temporal codec')
        a = self.arrays
        values = input_vector(problem, a['matrix_keys'])[a['feature_indices']]
        return ((values-a['feature_mean'])/a['feature_scale']).astype(np.float32)

    def encode(self, values):
        a = self.arrays
        return (((values-a['target_mean'])/a['target_scale']) @ a['components'].T)/a.get('latent_scale',1.)

    def decode(self, latent):
        a = self.arrays
        return ((latent*a.get('latent_scale',1.)) @ a['components'])*a['target_scale']+a['target_mean']

    def save(self, path):
        if Path(path).exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **self.arrays, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls({k:data[k] for k in data.files if k != 'metadata'}, json.loads(str(data['metadata'])))


def fit_codec(trajectories, train_ids, *, rank=32, max_features=128, seed=1, metadata=None):
    ids = list(map(int, train_ids))
    if len(ids) < 2 or len(set(ids)) != len(ids) or rank < 1 or max_features < 1:
        raise ValueError('At least two unique training trajectories and positive dimensions required')
    if min(ids) < 0 or max(ids) >= len(trajectories):
        raise ValueError('Invalid training trajectory IDs')
    training = [item for i in ids for item in trajectories[i]]
    base = training[0][0]
    m, n = base[0].shape
    if any(p[0].shape != (m,n) or p[-1] != base[-1] for p, _, _ in training):
        raise ValueError('Training LP dimensions changed')
    # Only training matrices decide which matrix coefficients become features.
    keys = set()
    for p, _, _ in training:
        diff = (p[0]-base[0]).tocoo()
        keys.update((diff.row*n+diff.col).tolist())
    matrix_keys = np.array(sorted(keys), dtype=np.int64)
    raw = np.stack([input_vector(p, matrix_keys) for p, _, _ in training])
    mean, scale = raw.mean(0), raw.std(0)
    variable = np.flatnonzero(scale > 1e-7)
    if not len(variable):
        variable = np.array([0])
    # Deduplicate normalized feature trajectories before taking largest spread.
    # Quantization affects proposals only; all original LP coefficients remain
    # intact in the corrector and certificate.
    unique, fingerprints = [], set()
    for index in variable[np.argsort(-scale[variable], kind='stable')]:
        normalized = np.round((raw[:,index]-mean[index])/max(scale[index],1e-3), 5)
        fingerprint = normalized.tobytes()
        if fingerprint not in fingerprints:
            unique.append(int(index)); fingerprints.add(fingerprint)
        if len(unique) >= max_features:
            break
    selected = np.array(sorted(unique), dtype=np.int64)
    target = np.stack([np.concatenate((x,y)) for _, x, y in training])
    target_mean = target.mean(0)
    target_scale = np.maximum(np.sqrt(np.mean(target**2, axis=0)), 1.)
    centered = (target-target_mean)/target_scale
    components = randomized_components(centered, rank, seed)
    latent_scale = np.maximum((centered @ components.T).std(0),1e-3)
    arrays = dict(matrix_keys=matrix_keys, feature_indices=selected,
        feature_mean=mean[selected], feature_scale=np.maximum(scale[selected],1e-3),
        target_mean=target_mean, target_scale=target_scale, components=components,
        latent_scale=latent_scale)
    info = dict(metadata or {}, dimensions=[n,m,int(base[-1])],
        rank=int(len(components)), feature_dim=2*len(selected)+1,
        train_trajectory_ids=ids, seed=int(seed), fit_scope='training trajectories only',
        target='original primal x and original row dual y; c-A.T@y convention',
        latent_standardized=True)
    return TemporalLPCodec(arrays, info)


def trajectory_arrays(trajectories, codec):
    base = np.stack([[codec.base_features(p) for p, _, _ in trajectory] for trajectory in trajectories])
    target = np.stack([[np.concatenate((x,y)) for _,x,y in trajectory] for trajectory in trajectories])
    latent = codec.encode(target).astype(np.float32)
    previous = np.zeros_like(latent); previous[:,1:] = latent[:,:-1]
    delta = np.zeros_like(base); delta[:,1:] = np.diff(base,axis=1)
    available = np.ones((*base.shape[:2],1),dtype=np.float32); available[:,0] = 0
    return np.concatenate((base,delta,available),axis=2), previous, latent, target
