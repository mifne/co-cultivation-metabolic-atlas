"""CPU execution of the SAME candidate/certificate algorithm for benchmarks.

This is not an online GPU fallback. It keeps the same basis, low-rank updates,
two refinements and residual/KKT gates as GpuBasisEvaluator, using NumPy/SciPy.
"""
import numpy as np

from src.gpu_certified_basis import GpuBasisEvaluator


class CpuBasisComparator(GpuBasisEvaluator):
    def __init__(self,anchor,variable_rows,primal_tolerance=1e-5,dual_tolerance=1e-7,gap_tolerance=1e-7):
        self.cp=np
        self.anchor=anchor
        self.primal_tolerance,self.dual_tolerance,self.gap_tolerance=primal_tolerance,dual_tolerance,gap_tolerance
        self.variable_rows=np.asarray(sorted(set(variable_rows)),dtype=int)
        self.host_a=anchor["lp"].a
        self.a=self.host_a
        self.inverse=anchor["inverse"]
        self.inverse_transpose=self.inverse.T.tocsr()
        self.a_transpose=self.a.T.tocsr()
        self.basic=anchor["basic"]
        self.active=anchor["active"]
        self.kind=anchor["kind"]
        self.var=self.variable_rows
        lookup={row:i for i,row in enumerate(self.active)}
        self.updated_positions=np.array([j for j,row in enumerate(self.var) if row in lookup],dtype=int)
        self.updated_active_rows=np.array([lookup[self.var[j]] for j in self.updated_positions],dtype=int)
        self.update_positions=self.updated_positions
        self.update_active=self.updated_active_rows
        self.u=self.inverse[:,self.update_active].toarray()
        self.neq=anchor["lp"].neq
        self.cpu_lp_calls=0

    def evaluate_host(self,**kwargs):
        result=super().evaluate_device(**kwargs)
        result["scope"]="CPU NumPy/SciPy execution of identical compiled-basis numerical certificate"
        return result
