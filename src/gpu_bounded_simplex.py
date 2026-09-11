"""Experimental FP64 GPU bounded-variable two-phase simplex.

Matrix pivots/refactorization, pricing and feasibility arithmetic use CuPy.
Host control chooses pivot indices; no CPU LP optimizer is called. Intended
as an independently checked alternative for poorly conditioned cuOpt LPs.
"""
import time
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix, vstack


class GpuBoundedSimplex:
    method = "gpu_bounded_simplex"
    name = "gpu_bounded_simplex_experimental"

    def __init__(self, tolerance=1e-8, time_limit=120, max_iterations=200000, **ignored):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.cp, self.gpu_csr = cp, gpu_csr
        self.tolerance, self.residual_tolerance = tolerance, 1e-5
        self.time_limit = time_limit
        self.max_iterations = int(max_iterations)
        self.log_to_console = bool(ignored.get("log_to_console",False))
        self.history = []
        self.failure_snapshot = None
        self.numerical_repair_basis = None
        self.basis_cache = {}
        self.last_basis = {"basis":[],"upper":[]}
        self.pivot_kernel = cp.RawKernel(r'''
        extern "C" __global__ void pivot(double* a, const double* col,
             const double* row, long long size, int width, int pivotrow) {
          long long k=(long long)blockDim.x*blockIdx.x+threadIdx.x;
          if(k<size) {int i=k/width,j=k%width;
            a[k] = i==pivotrow ? row[j] : a[k]-col[i]*row[j];}
        }''', "pivot")

    def solve(self,c,**kwargs):
        """Restore primal feasibility on GPU before returning a numerical optimum.

        Harris pivots/refactorizations can expose new bound infeasibilities.
        Rebuild Phase I from that candidate basis; never accept or clip the
        infeasible vector, and never change the LP or invoke a CPU optimizer.
        """
        started=time.perf_counter()
        original_limit=self.time_limit
        history_start=len(self.history)
        try:
            for attempt in range(4):
                self.time_limit=max(1e-9,original_limit-(time.perf_counter()-started))
                self.numerical_repair_basis=None
                result=self._solve_once(c,**kwargs)
                record=self.history[-1]
                if result.success:
                    self.failure_snapshot=None
                    break
                if (record["status"]!="optimal" or self.numerical_repair_basis is None or
                        time.perf_counter()-started>=original_limit):
                    break
                if attempt<3:
                    self.basis_cache[kwargs.get("stage_key","default")]=self.numerical_repair_basis
                    self.last_basis=self.numerical_repair_basis
        finally:
            self.time_limit=original_limit
        attempts=[dict(record) for record in self.history[history_start:]]
        record=dict(attempts[-1])
        record.update(gpu_lp_attempts=len(attempts),gpu_primal_repairs=len(attempts)-1,
            min_pivot=min(r["min_pivot"] for r in attempts),
            iteration_limit_per_attempt=self.max_iterations,
            crash_setup_seconds_total=sum(r["host_qr_crash_seconds"] for r in attempts),
            iterations=sum(r["iterations"] for r in attempts),
            phase_iterations=[sum(r["phase_iterations"][i] if len(r["phase_iterations"])>i else 0
                                  for r in attempts) for i in range(2)],
            worst_attempt_residual=max(r["max_original_residual"] for r in attempts),
            total_seconds=time.perf_counter()-started,time_limit=original_limit)
        if len(attempts)>1: record["primal_repair_attempt_records"]=attempts
        self.history[history_start:]=[record]
        result.message=str(record)
        if self.log_to_console: print(record,flush=True)
        return result

    def _solve_once(self,c,*,A_eq=None,b_eq=None,A_ub=None,b_ub=None,bounds=None,**ignored):
        cp = self.cp
        start = time.perf_counter()
        c = np.asarray(c,float); n=len(c)
        eq=csr_matrix((0,n)) if A_eq is None else csr_matrix(A_eq)
        ub=csr_matrix((0,n)) if A_ub is None else csr_matrix(A_ub)
        original=vstack((eq,ub),format="csr")
        rhs=np.r_[[] if b_eq is None else b_eq, [] if b_ub is None else b_ub]
        bounds=[(0,None)]*n if bounds is None else bounds
        lower=np.array([-np.inf if a is None else a for a,b in bounds])
        upper=np.array([np.inf if b is None else b for a,b in bounds])
        shift=np.where(np.isfinite(lower),lower,np.where(np.isfinite(upper),upper,0.))
        signs=np.where(np.isfinite(lower),1.,-1.)
        free=np.flatnonzero(~np.isfinite(lower)&~np.isfinite(upper))
        signs[free]=1.
        dense=cp.asarray(original.toarray())
        transformed=cp.concatenate((dense*cp.asarray(signs),-dense[:,free]),axis=1)
        costs=cp.concatenate((cp.asarray(c*signs),cp.asarray(-c[free])))
        widths=np.where(np.isfinite(lower)&np.isfinite(upper),upper-lower,np.inf)
        limits=cp.asarray(np.r_[widths,np.full(len(free),np.inf)])
        b=cp.asarray(rhs)-dense@cp.asarray(shift)
        m=len(rhs); nz=transformed.shape[1]; ni=ub.shape[0]
        orientation=cp.where(b<0,-1.,1.)
        slack=cp.zeros((m,ni),dtype=cp.float64)
        if ni: slack[cp.arange(eq.shape[0],m),cp.arange(ni)]=1.
        matrix=cp.concatenate((transformed,slack),axis=1)*orientation[:,None]
        row_scale=cp.maximum(cp.max(cp.abs(matrix),axis=1),1.)
        matrix/=row_scale[:,None]; b=cp.abs(b)/row_scale
        limits=cp.r_[limits,cp.full(ni,cp.inf)]
        column_ids=ignored.get("column_ids") or [f"v{i}" for i in range(n)]
        inequality_ids=ignored.get("inequality_ids") or [f"r{i}" for i in range(ni)]
        names=column_ids+["negative:"+column_ids[i] for i in free]+["slack:"+i for i in inequality_ids]
        stage_key=ignored.get("stage_key", "default")
        previous=self.basis_cache.get(stage_key,self.last_basis)
        # Algebraic QR crash, not CPU LP optimization. Prefer the preceding
        # optimal basis by persistent presolved column names, then repair it.
        qr_start=time.perf_counter()
        from scipy.linalg import qr
        host_matrix=matrix.get()
        previous_names=set(previous["basis"])
        preferred=np.array([i for i,name in enumerate(names) if name in previous_names],dtype=int)
        threshold=max(host_matrix.shape)*np.finfo(float).eps
        if len(preferred):
            q,r,piv=qr(host_matrix[:,preferred],mode="full",pivoting=True,check_finite=False)
            rank_pre=int(np.count_nonzero(np.abs(np.diag(r))>threshold))
            chosen=preferred[piv[:rank_pre]]
            if rank_pre<m:
                # Only the missing orthogonal directions need new columns.
                residual=q[:,rank_pre:].T@host_matrix
                r_more,p_more=qr(residual,mode="r",pivoting=True,check_finite=False)
                rank_more=int(np.count_nonzero(np.abs(np.diag(r_more))>threshold))
                chosen=np.r_[chosen,p_more[:rank_more]]
        else:
            r,piv=qr(host_matrix,mode="r",pivoting=True,check_finite=False)
            rank=int(np.count_nonzero(np.abs(np.diag(r))>threshold))
            chosen=piv[:rank]
        if len(chosen)<m:
            _,row_piv=qr(host_matrix[:,chosen].T,mode="r",pivoting=True,check_finite=False)
            selected=np.sort(row_piv[:len(chosen)])
            matrix=cp.ascontiguousarray(matrix[selected]); b=b[selected]; m=len(chosen)
        basis=cp.asarray(chosen,dtype=cp.int64)
        x=cp.zeros(nz+ni)
        previous_upper=set(previous["upper"])
        host_limits=limits.get()
        upper_ids=[i for i,name in enumerate(names) if name in previous_upper and np.isfinite(host_limits[i])]
        if upper_ids: x[cp.asarray(upper_ids)]=limits[cp.asarray(upper_ids)]
        x[basis]=0.
        B=matrix[:,basis]
        x[basis]=cp.linalg.solve(B,b-matrix@x)
        violated=(x[basis]<-1e-9)|(x[basis]>limits[basis]+1e-9)
        bad=cp.flatnonzero(violated)
        bad_count=len(bad)
        art_start=nz+ni
        signs_art=cp.where(x[basis[bad]]<0,-1.,1.)
        artificial_columns=B[:,bad]*signs_art[None,:]
        artificial_values=cp.where(signs_art<0,-x[basis[bad]],x[basis[bad]]-limits[basis[bad]])
        x[basis[bad]]=cp.where(signs_art<0,0.,limits[basis[bad]])
        matrix=cp.ascontiguousarray(cp.concatenate((matrix,artificial_columns),axis=1))
        limits=cp.r_[limits,cp.full(bad_count,cp.inf)]
        x=cp.r_[x,artificial_values]
        basis[bad]=cp.arange(art_start,art_start+bad_count)
        total=matrix.shape[1]
        tableau=cp.ascontiguousarray(cp.linalg.solve(matrix[:,basis],matrix))
        qr_seconds=time.perf_counter()-qr_start
        iterations=0; phase_counts=[]; status="unknown"
        min_pivot=1.; stalls=0

        def refactor():
            nonlocal tableau,x
            B=matrix[:,basis]
            tableau=cp.ascontiguousarray(cp.linalg.solve(B,matrix))
            # Correct accumulated pivot/solve roundoff on the device. A small
            # scaled-tableau error can be large in the original GEM units.
            tableau += cp.linalg.solve(B,matrix-B@tableau)
            nb=x.copy(); nb[basis]=0.
            basis_rhs=b-matrix@nb
            xb_refined=cp.linalg.solve(B,basis_rhs)
            xb_refined += cp.linalg.solve(B,basis_rhs-B@xb_refined)
            x[basis]=xb_refined

        for phase in (1,2):
            objective=cp.zeros(total)
            if phase==1: objective[art_start:]=1.
            else:
                objective[:nz]=costs
                limits[art_start:]=0.
            reduced=objective-objective[basis]@tableau
            phase_start=iterations
            freshly_refactored=False
            while iterations < self.max_iterations and time.perf_counter()-start < self.time_limit:
                basic=cp.zeros(total,dtype=bool); basic[basis]=True
                movable=limits>self.tolerance
                candidates=cp.where(~basic & movable,
                    cp.where(x<=self.tolerance,-reduced,
                             cp.where(x>=limits-self.tolerance,reduced,-cp.inf)),-cp.inf)
                j=int(cp.argmax(candidates).get())
                if float(candidates[j].get()) <= self.tolerance:
                    if not freshly_refactored:
                        refactor()
                        reduced=objective-objective[basis]@tableau
                        freshly_refactored=True
                        continue
                    status="optimal"; break
                freshly_refactored=False
                # Switch to Bland pricing after repeated degenerate pivots.
                if stalls > 100:
                    j=int(cp.flatnonzero(candidates>self.tolerance)[0].get())
                direction=1. if float(reduced[j].get())<0 else -1.
                col=tableau[:,j].copy(); movement=col*direction
                xb=x[basis]
                distance=cp.where(movement>0,cp.maximum(xb,0),cp.maximum(limits[basis]-xb,0))
                ratio=cp.where(cp.abs(movement)>1e-9,distance/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                relaxed=cp.where(cp.abs(movement)>1e-9,(distance+1e-8)/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                relaxed_min=float(cp.min(relaxed).get())
                # Harris two-pass ratio: favour a stable pivot among limiting rows.
                i=int(cp.argmax(cp.where(ratio<=relaxed_min,cp.abs(col),-1.)).get())
                basic_step=float(ratio[i].get())
                own_step=float((limits[j]-x[j] if direction>0 else x[j]).get())
                delta=min(basic_step,own_step)
                if not np.isfinite(delta): status="unbounded"; break
                stalls=stalls+1 if delta<1e-10 else 0
                x[basis]=xb-movement*delta
                x[j]+=direction*delta
                if basic_step <= own_step:
                    leaving=int(basis[i].get())
                    x[leaving]=0. if float(movement[i].get())>0 else limits[leaving]
                    pivot=float(col[i].get()); min_pivot=min(min_pivot,abs(pivot))
                    row=tableau[i].copy()/pivot
                    self.pivot_kernel(((tableau.size+255)//256,),(256,),
                        (tableau,col,row,np.int64(tableau.size),np.int32(total),np.int32(i)))
                    reduced-=reduced[j]*row
                    basis[i]=j
                else:
                    x[j]=limits[j] if direction>0 else 0.
                iterations+=1
                if iterations % 100 == 0:
                    refactor()
                    reduced=objective-objective[basis]@tableau
            else: status="limit"
            phase_counts.append(iterations-phase_start)
            if status != "optimal": break
            if phase==1:
                artificial=float(x[art_start:].sum().get())
                if artificial > 1e-6: status="phase_one_infeasible_or_inaccurate"; break
                # Recompute the basis before changing objectives.
                refactor()
        values=cp.asarray(shift)+cp.asarray(signs)*x[:n]
        if len(free): values[cp.asarray(free)]-=x[n:n+len(free)]
        activity=dense@values
        residuals=cp.r_[cp.abs(activity[:eq.shape[0]]-cp.asarray(rhs[:eq.shape[0]])),
                       cp.maximum(activity[eq.shape[0]:]-cp.asarray(rhs[eq.shape[0]:]),0),
                       cp.maximum(cp.asarray(lower)-values,0),cp.maximum(values-cp.asarray(upper),0)]
        residual=float(residuals.max().get())
        success=status=="optimal" and len(phase_counts)==2 and residual <= self.residual_tolerance
        record=dict(method=self.method,status=status,success=success,iterations=iterations,
            iteration_limit=self.max_iterations, time_limit=self.time_limit,
            phase_iterations=phase_counts,min_pivot=min_pivot,max_original_residual=residual,
            max_artificial_value=float(cp.max(cp.abs(x[art_start:])).get()) if bad_count else 0.,
            objective=float(cp.asarray(c)@values),total_seconds=time.perf_counter()-start,
            cpu_lp_calls=0,tableau_shape=list(tableau.shape),host_qr_crash_seconds=qr_seconds,
            crash_artificials=bad_count,warm_basis_available=bool(previous["basis"]))
        record.update(max_equation_residual=float(cp.max(cp.abs(activity[:eq.shape[0]]-cp.asarray(rhs[:eq.shape[0]]))).get()) if eq.shape[0] else 0.,
            max_inequality_violation=float(cp.max(cp.maximum(activity[eq.shape[0]:]-cp.asarray(rhs[eq.shape[0]:]),0)).get()) if ub.shape[0] else 0.,
            max_lower_violation=float(cp.max(cp.maximum(cp.asarray(lower)-values,0)).get()),
            max_upper_violation=float(cp.max(cp.maximum(values-cp.asarray(upper),0)).get()))
        if success:
            self.basis_cache[stage_key]={"basis":[names[i] for i in basis.get() if i<art_start],
                "upper":[names[i] for i in cp.flatnonzero(cp.isfinite(limits[:art_start]) &
                    (cp.abs(x[:art_start]-limits[:art_start])<1e-7)).get()]}
            self.last_basis=self.basis_cache[stage_key]
        else:
            # Preserve the exact failed LP and incoming warm start for a fast
            # numerical regression; never accept it or replace it with CPU.
            import json
            self.numerical_repair_basis={"basis":[names[i] for i in basis.get() if i<art_start],
                "upper":[names[i] for i in cp.flatnonzero(cp.isfinite(limits[:art_start]) &
                    (cp.abs(x[:art_start]-limits[:art_start])<1e-7)).get()]}
            self.failure_snapshot=dict(c=c, data=original.data, indices=original.indices,
                indptr=original.indptr, shape=np.asarray(original.shape), rhs=rhs,
                neq=np.asarray(eq.shape[0]), lower=lower, upper=upper,
                failed_values=values.get(),
                repair_basis_json=np.asarray(json.dumps(self.numerical_repair_basis)),
                metadata_json=np.asarray(json.dumps(dict(column_ids=column_ids,
                    inequality_ids=inequality_ids, stage_key=stage_key, previous=previous))))
        self.history.append(record)
        return SimpleNamespace(success=success,x=values.get() if success else None,
                               fun=record["objective"],message=str(record))
