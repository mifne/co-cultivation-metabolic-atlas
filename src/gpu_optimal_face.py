"""GPU secondary LP via reduced-cost fixing, certified against original LP.

The restricted-face solve is only a candidate generator. A reconstructed dual
and full numerical KKT test must also certify the ORIGINAL secondary problem
including its allowed primary-objective loss. Neither CPU reference nor its
three objectives are changed. No Gurobi dependency is used.
"""


def solve_on_primary_face(solver,primary,inputs,secondary_c,allowance,fix_tolerance=1e-8,pivot_budget=None):
    cp=solver.cp;n=solver.n
    state=primary["warm_state"]
    lower,upper=state["lower"],state["upper"]
    same_bounds=(inputs["lower"]==lower[:,:n]).all(axis=1)&(inputs["upper"]==upper[:,:n]).all(axis=1)
    rp=state["reduced"]
    nonzero=cp.abs(rp*state["scale"])>fix_tolerance
    target=cp.where(rp>=0,lower,upper)
    fixed=nonzero&cp.isfinite(target)
    face_lower=cp.where(fixed,target,lower)
    face_upper=cp.where(fixed,target,upper)
    secondary=solver.run_device(**dict(inputs,c=secondary_c),warm_start=state,
        full_lower=face_lower,full_upper=face_upper,pivot_budget=pivot_budget)
    chosen=secondary["warm_state"]
    x,y2,r2=chosen["x"],chosen["y"],chosen["reduced"]
    # A nonnegative multiplier turns the restricted-face dual into a dual of
    # [original constraints; primary_cost*x <= primary_optimum+allowance].
    ratio=cp.where(fixed&((rp*r2)<0),-r2/cp.where(nonzero,rp,1.),0.)
    multiplier=cp.maximum(cp.max(ratio,axis=1),0.)*(1.+1e-10)+1e-12
    y=y2+multiplier[:,None]*state["y"]
    activity=(solver.matrix@x.T).T
    activity[:,solver.var]+=cp.einsum("bkn,bn->bk",inputs["delta"],x[:,:n])
    transpose=(solver.matrix_transpose@y.T).T
    transpose[:,:n]+=cp.einsum("bkn,bk->bn",inputs["delta"],y[:,solver.var])
    reduced=chosen["cost"]-transpose+multiplier[:,None]*state["cost"]
    original_primary=cp.sum(state["cost"]*state["x"],axis=1)
    primary_value=cp.sum(state["cost"]*x,axis=1)
    primary_bound=original_primary+allowance
    equation=cp.max(cp.abs(activity)/inputs["row_scale"],axis=1)
    bounds=cp.maximum(cp.max(cp.maximum(lower-x,x-upper)/state["scale"],axis=1),0.)
    primal=cp.maximum(cp.maximum(equation,bounds),cp.maximum(primary_value-primary_bound,0.))
    original_rc=reduced*state["scale"]
    dual=cp.maximum(cp.max(cp.where(~cp.isfinite(lower),original_rc,0.),axis=1),
        cp.max(cp.where(~cp.isfinite(upper),-original_rc,0.),axis=1))
    selected=cp.where(reduced>=0,lower,upper)
    complementarity=cp.sum(cp.abs(reduced*(x-cp.where(cp.isfinite(selected),selected,x))),axis=1)
    allowance_gap=cp.abs(multiplier*(primary_bound-primary_value))
    gap=complementarity+allowance_gap
    value=cp.sum(chosen["cost"]*x,axis=1)
    gap/=cp.maximum(1.,cp.abs(value))
    accepted=primary["accepted"]&secondary["accepted"]&same_bounds
    accepted&=cp.isfinite(primal)&cp.isfinite(dual)&cp.isfinite(gap)&cp.isfinite(multiplier)
    accepted&=(primal<=solver.primal_tolerance)&(dual<=solver.dual_tolerance)&(gap<=solver.gap_tolerance)
    accepted&=(primary_value<=primary_bound+1e-8)&(allowance>=0)
    return dict(accepted=accepted,values=cp.where(accepted[:,None],x[:,:n]/inputs["col_scale"],cp.nan),
        objective=cp.where(accepted,value,cp.nan),primal_residual=primal,dual_violation=dual,relative_kkt_gap=gap,
        primary_value=primary_value,primary_optimum=original_primary,multiplier=multiplier,
        fixed_variables=cp.sum(fixed,axis=1),pivots=secondary["pivots"],cpu_lp_calls=0,
        complementarity_gap=complementarity/cp.maximum(1.,cp.abs(value)),
        allowance_gap=allowance_gap/cp.maximum(1.,cp.abs(value)),
        scope="GPU reduced-cost face candidate + original allowed-loss LP numerical KKT certificate")
