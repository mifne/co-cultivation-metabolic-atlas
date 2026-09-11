"""Conservative exact row-dependency removal, proposed by bounded CPU QR.

Numerical rank only proposes rows and rational coefficients. A row is removed
only after Fraction.from_float arithmetic proves EVERY stored coefficient and
the RHS identity exactly. Sparse float64 dual compression additionally requires
exactly representable (dyadic) proof weights. Near dependencies stay in the LP.
This is CPU algebraic setup, not an LP solve, numerical tolerance relaxation,
reference-solution lookup, or an original-LP optimality certificate.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time

import numpy as np
from scipy.linalg import qr,solve_triangular
from scipy.sparse import csr_matrix

from .gpu_pdhg_corrector import _validated_problem
from .lp_trace import problem_hash
from .lp_zero_face import _freeze_problem,_readonly


@dataclass(frozen=True)
class ExactEqualityProof:
    row: int
    basis_rows: tuple
    coefficients: tuple
    kind: str = 'qr_proposed_exact_binary64_identity'

    def as_dict(self):
        return dict(row=self.row,basis_rows=list(self.basis_rows),kind=self.kind,
            coefficients=[dict(numerator=c.numerator,denominator=c.denominator)
                          for c in self.coefficients])


def _hash_arrays(items):
    digest=hashlib.sha256()
    for name,value,dtype in items:
        array=np.asarray(value,dtype=dtype)
        digest.update(name.encode('ascii'))
        digest.update(np.asarray(array.shape,dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def equality_fingerprint(problem):
    """Exact equality matrix/RHS identity; independent of bounds, cost and UB rows."""
    a,rhs,_lo,_hi,_c,neq=problem
    e=a[:neq].tocsr()
    return _hash_arrays((('shape',e.shape,'<i8'),('indptr',e.indptr,'<i8'),
        ('indices',e.indices,'<i8'),('data',e.data,'<f8'),('rhs',rhs[:neq],'<f8'),
        ('neq',[neq],'<i8')))


def _sparse_fingerprint(matrix):
    return _hash_arrays((('shape',matrix.shape,'<i8'),('indptr',matrix.indptr,'<i8'),
        ('indices',matrix.indices,'<i8'),('data',matrix.data,'<f8')))


def _proof_fingerprint(proofs):
    return hashlib.sha256(json.dumps([p.as_dict() for p in proofs],
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


class ExactEqualityReduction:
    """Immutable LP snapshots and exact stored-matrix row/dual maps.

    ``rows`` maps retained rows to original rows, in original order, followed
    by all untouched inequality rows. ``dual_compression`` is P with shape
    [reduced_rows, original_rows]; over exact arithmetic its stored entries
    satisfy A_original.T = A_reduced.T P and b_original = P.T b_reduced.
    Floating-point applications still require the full original LP gate.
    Removed duals lift to zero; compressed removed duals contribute only to
    permanently retained QR-basis rows, never another removed row.

    ``reuse_from`` avoids QR when the exact equality fingerprint matches.
    Mismatches raise explicitly. Changed bounds/cost/inequalities are copied
    into the new snapshot and do not invalidate an equality-only proof.
    """
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):
            raise AttributeError('ExactEqualityReduction is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,problem,*,qr_threshold=1e-12,max_denominator=1_000_000,
                 max_dimension=6000,max_memory_mb=1024.,max_candidates=128,
                 max_support=256,reuse_from=None):
        started=time.perf_counter()
        if (type(max_dimension) is not int or not 1<=max_dimension<=6000
                or type(max_candidates) is not int or not 1<=max_candidates<=128
                or type(max_support) is not int or not 1<=max_support<=1024
                or type(max_denominator) is not int or not 1<=max_denominator<=1_000_000_000
                or not math.isfinite(max_memory_mb) or max_memory_mb<=0.
                or not math.isfinite(qr_threshold) or not 0.<qr_threshold<1.):
            raise ValueError('Invalid bounded exact-equality setup configuration')
        original=_freeze_problem(_validated_problem(problem))
        a,rhs,lo,hi,c,neq=original
        self.original=original
        self.original_hash=problem_hash(original)
        self.equality_fingerprint=equality_fingerprint(original)
        counts=dict(numerical_candidate_count=0,examined_candidate_count=0,
            exact_verification_attempts=0,matrix_identity_rejections=0,
            rhs_identity_rejections=0,nonrepresentable_weight_rejections=0,
            support_guard_skips=0,unverified_candidates_retained=0,
            nominated_qr_rank=None,qr_seconds=0.,proof_seconds=0.,
            estimated_dense_working_bytes=0,reused_proofs=reuse_from is not None)
        if reuse_from is not None:
            if not isinstance(reuse_from,ExactEqualityReduction):
                raise ValueError('reuse_from must be an ExactEqualityReduction')
            reuse_from.validate_integrity()
            if reuse_from.equality_fingerprint!=self.equality_fingerprint:
                raise ValueError('Exact equality fingerprint mismatch for proof reuse')
            proofs=reuse_from.proofs
            counts.update(nominated_qr_rank=reuse_from.summary['nominated_qr_rank'],
                reused_from_proof_fingerprint=reuse_from.proof_fingerprint)
        else:
            proofs,counts=self._discover(original,qr_threshold,max_denominator,
                max_dimension,max_memory_mb,max_candidates,max_support,counts)
        self.proofs=tuple(sorted(proofs,key=lambda p:p.row))
        removed={p.row for p in self.proofs}
        if any(r<0 or r>=neq for r in removed) or any(
                r in removed or not 0<=r<neq for p in self.proofs for r in p.basis_rows):
            raise ValueError('Proof rows must depend only on retained equalities')
        self.removed_rows=_readonly(np.array(sorted(removed),dtype=np.int64))
        self.rows=_readonly(np.array([r for r in range(a.shape[0]) if r not in removed],dtype=np.int64))
        self.reduced=_freeze_problem((a[self.rows].tocsr(),rhs[self.rows],lo,hi,c,
                                     int(np.sum(self.rows<neq))))
        self.reduced_hash=problem_hash(self.reduced)
        reduced_index={int(row):i for i,row in enumerate(self.rows)}
        rr=list(range(len(self.rows)));cc=self.rows.tolist();vv=[1.]*len(self.rows)
        for proof in self.proofs:
            for row,weight in zip(proof.basis_rows,proof.coefficients):
                numeric=float(weight)
                if not np.isfinite(numeric) or Fraction.from_float(numeric)!=weight:
                    raise ValueError('Compression weight is not exactly representable in binary64')
                rr.append(reduced_index[row]);cc.append(proof.row);vv.append(numeric)
        compression=csr_matrix((vv,(rr,cc)),shape=(len(self.rows),a.shape[0]),dtype=np.float64)
        compression.sum_duplicates();compression.sort_indices()
        for vector in (compression.data,compression.indices,compression.indptr):
            vector.flags.writeable=False
        self.dual_compression=compression
        self.proof_fingerprint=_proof_fingerprint(self.proofs)
        self.compression_fingerprint=_sparse_fingerprint(compression)
        self._rows_fingerprint=_hash_arrays((('rows',self.rows,'<i8'),('removed',self.removed_rows,'<i8')))
        self._summary=dict(counts,original_rows=a.shape[0],original_equalities=neq,
            reduced_rows=len(self.rows),reduced_equalities=self.reduced[-1],
            removed_rows=len(self.removed_rows),removed_zero_equalities=sum(
                p.kind=='exact_zero_equality' for p in self.proofs),
            proof_fingerprint=self.proof_fingerprint,equality_fingerprint=self.equality_fingerprint,
            compression_fingerprint=self.compression_fingerprint,
            original_hash=self.original_hash,reduced_hash=self.reduced_hash,
            qr_threshold=float(qr_threshold),max_denominator=max_denominator,
            max_support=max_support,max_candidates=max_candidates,
            max_dimension=max_dimension,max_memory_mb=float(max_memory_mb),
            setup_seconds=time.perf_counter()-started,cpu_lp_calls=0,gpu_calls=0,
            rows_removed_by_numerical_tolerance_alone=0,
            exact_identity_scope='Fraction.from_float for every coefficient and RHS; proof weights exactly representable in binary64',
            completeness_claimed=False,original_certificate_still_required=True)
        self._sealed=True

    @staticmethod
    def _discover(problem,threshold,max_denominator,max_dimension,max_memory_mb,
                  max_candidates,max_support,counts):
        a,rhs,_lo,_hi,_c,neq=problem
        e=a[:neq].tocsr()
        nonzero=np.flatnonzero(np.diff(e.indptr))
        zero=np.flatnonzero(np.diff(e.indptr)==0)
        proofs=[ExactEqualityProof(int(row),(),(),'exact_zero_equality')
                for row in zero if rhs[row]==0.]
        if not len(nonzero):
            counts['nominated_qr_rank']=0
            return proofs,counts
        m,n=len(nonzero),e.shape[1]
        if max(m,n)>max_dimension:
            raise ValueError('Equality dimensions exceed the configured dense-QR guard')
        small=min(m,n)
        estimate=(8*(4*m*n+3*small*small+3*n*max_candidates)
                  +4*(e.data.nbytes+e.indices.nbytes+e.indptr.nbytes))
        if estimate>max_memory_mb*1024**2:
            raise ValueError('Estimated dense-QR working memory exceeds guard')
        counts['estimated_dense_working_bytes']=int(estimate)
        normalized=e[nonzero].copy()
        divisor=np.asarray(abs(normalized).max(axis=1).toarray()).ravel()
        with np.errstate(over='ignore',under='ignore'):
            normalized.data/=np.repeat(divisor,np.diff(normalized.indptr))
        if not np.isfinite(normalized.data).all() or np.any(normalized.data==0.):
            raise ValueError('Row normalization loses nonzero data or creates nonfinite values')
        before=time.perf_counter()
        r,pivots=qr(normalized.T.toarray(order='F'),mode='r',pivoting=True,
                    overwrite_a=True,check_finite=False)
        counts['qr_seconds']=time.perf_counter()-before
        r=r[:small]
        if not np.isfinite(r).all():raise ValueError('QR returned nonfinite coefficients')
        diagonal=np.abs(np.diag(r))
        relative=diagonal/diagonal[0] if diagonal[0]>0. else np.zeros_like(diagonal)
        rank=int(np.count_nonzero(relative>threshold))
        if np.any(relative[:rank]<=threshold):
            raise ValueError('Non-prefix QR rank proposal is ambiguous')
        candidates=pivots[rank:rank+max_candidates]
        basis=pivots[:rank]
        counts.update(nominated_qr_rank=rank,numerical_candidate_count=len(pivots)-rank,
                      examined_candidate_count=len(candidates))
        coefficients=(solve_triangular(r[:rank,:rank],r[:rank,rank:rank+len(candidates)],
            lower=False,check_finite=False) if rank else np.zeros((0,len(candidates))))
        before=time.perf_counter()
        exact_rows={}
        def row_values(row):
            if row not in exact_rows:
                start,end=e.indptr[row:row+2]
                exact_rows[row]={int(j):Fraction.from_float(float(v)) for j,v in
                    zip(e.indices[start:end],e.data[start:end])}
            return exact_rows[row]
        def verify(row,support,weights):
            counts['exact_verification_attempts']+=1
            reconstructed={}
            for source,weight in zip(support,weights):
                for column,value in row_values(source).items():
                    reconstructed[column]=reconstructed.get(column,Fraction(0))+weight*value
            reconstructed={j:v for j,v in reconstructed.items() if v}
            if reconstructed!=row_values(row):
                counts['matrix_identity_rejections']+=1
                return False
            predicted=sum((weight*Fraction.from_float(float(rhs[source]))
                           for source,weight in zip(support,weights)),Fraction(0))
            if predicted!=Fraction.from_float(float(rhs[row])):
                counts['rhs_identity_rejections']+=1
                return False
            return True
        denominators=sorted({d for d in (1,2,4,8,16,64,256,1024,max_denominator)
                             if d<=max_denominator})
        for column,candidate in enumerate(candidates):
            row=int(nonzero[candidate])
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                proposal=coefficients[:,column]*(divisor[candidate]/divisor[basis])
            if not np.isfinite(proposal).all():continue
            approximations=[Fraction.from_float(float(v)) for v in proposal]
            seen=set()
            for denominator in denominators:
                rational=tuple(v.limit_denominator(denominator) for v in approximations)
                if rational in seen:continue
                seen.add(rational)
                selected=[i for i,v in enumerate(rational) if v]
                if len(selected)>max_support:
                    counts['support_guard_skips']+=1
                    continue
                support=tuple(int(nonzero[basis[i]]) for i in selected)
                weights=tuple(rational[i] for i in selected)
                # Proof of exact dependence alone does not make a 1/3 weight
                # representable by the runtime's stored binary64 sparse map.
                if any(not math.isfinite(float(v)) or Fraction.from_float(float(v))!=v for v in weights):
                    counts['nonrepresentable_weight_rejections']+=1
                    continue
                if verify(row,support,weights):
                    proofs.append(ExactEqualityProof(row,support,weights))
                    break
        counts['unverified_candidates_retained']=counts['numerical_candidate_count']-sum(
            p.kind!='exact_zero_equality' for p in proofs)
        counts['proof_seconds']=time.perf_counter()-before
        return proofs,counts

    @property
    def summary(self):
        return dict(self._summary)

    def validate_integrity(self):
        if (problem_hash(self.original)!=self.original_hash
                or problem_hash(self.reduced)!=self.reduced_hash
                or equality_fingerprint(self.original)!=self.equality_fingerprint
                or _proof_fingerprint(self.proofs)!=self.proof_fingerprint
                or _sparse_fingerprint(self.dual_compression)!=self.compression_fingerprint
                or _hash_arrays((('rows',self.rows,'<i8'),('removed',self.removed_rows,'<i8')))
                   !=self._rows_fingerprint):
            raise ValueError('Exact-equality LP snapshot or proof map was modified')

    @staticmethod
    def _vector(value,length):
        array=np.asarray(value,dtype=np.float64)
        if array.shape!=(length,) or not np.isfinite(array).all():
            raise ValueError('Finite vector with the exact shape required')
        return array

    def compress_dual(self,original_y):
        self.validate_integrity()
        y=self._vector(original_y,len(self.original[1]))
        with np.errstate(over='raise',invalid='raise'):
            try:result=np.asarray(self.dual_compression@y).ravel()
            except FloatingPointError as error:raise ValueError('Dual compression overflowed') from error
        if not np.isfinite(result).all():raise ValueError('Dual compression is nonfinite')
        return result

    def lift_dual(self,reduced_y):
        self.validate_integrity()
        result=np.zeros(len(self.original[1]),dtype=np.float64)
        result[self.rows]=self._vector(reduced_y,len(self.rows))
        return result

    def lift_primal(self,x):
        self.validate_integrity()
        return self._vector(x,len(self.original[4])).copy()
