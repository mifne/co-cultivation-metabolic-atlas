"""CPU-only numerical/optimal-face audit of a captured, unchanged LP."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lp", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--net-metabolites", nargs="*", default=["so4_e", "nh4_e", "pi_e", "o2_e"])
    p.add_argument("--face-relative-tolerance", type=float, default=1e-8)
    args = p.parse_args()
    if args.face_relative_tolerance < 0: raise ValueError("Nonnegative face tolerance required")
    data = np.load(args.lp)
    meta = json.loads(args.metadata.read_text())
    matrix = csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
    neq = int(data["neq"])
    c = data["c"]
    captured_x = data["cpu_x"] if "cpu_x" in data else data["gpu_x"]
    kw = dict(A_eq=matrix[:neq], b_eq=data["rhs"][:neq], A_ub=matrix[neq:],
        b_ub=data["rhs"][neq:], bounds=list(zip(data["lower"], data["upper"])))
    critical = sorted(set([v[0] for v in meta["growth_terms"].values()] +
        [i for i,r in enumerate(meta["reaction_ids"]) if r.lower() in {"ex_pha_c", "ex_phv_c"}]))
    records = []
    for method, tolerance in (("highs-ds",None),("highs-ipm",None),("highs-ds",1e-9),("highs-ds",1e-10)):
        options = dict(presolve=True)
        if tolerance is not None:
            options.update(primal_feasibility_tolerance=tolerance,dual_feasibility_tolerance=tolerance)
        result = linprog(c, **kw, method=method, options=options)
        row = dict(method=method,tolerance=tolerance,success=result.success,message=result.message,
            objective=float(result.fun) if result.success else None)
        if result.success:
            row["critical_fluxes"] = {f"{meta['reaction_species'][i]}:{meta['reaction_ids'][i]}":float(result.x[i]) for i in critical}
            row["max_solution_difference_from_capture"] = float(np.max(np.abs(result.x-captured_x)))
            row["active_shared_rows"] = [dict(metabolite=name, rhs=float(kw["b_ub"][3+j]),
                slack=float(result.ineqlin.residual[3+j]), marginal=float(result.ineqlin.marginals[3+j]))
                for j,name in enumerate(sorted(meta["exchange_terms"]))
                if abs(result.ineqlin.marginals[3+j])>1e-8]
            if "stage_3" in args.lp.stem:
                allowance = args.face_relative_tolerance*max(1.,abs(float(result.fun)))
                face = dict(kw,A_ub=vstack((kw["A_ub"],csr_matrix(c[None])),format="csr"),
                    b_ub=np.r_[kw["b_ub"],result.fun+allowance])
                ranges=[]
                for i in critical:
                    objective=np.zeros(len(c));objective[i]=1.
                    lo=linprog(objective,**face,method=method,options=options)
                    hi=linprog(-objective,**face,method=method,options=options)
                    ranges.append(dict(species=meta["reaction_species"][i],reaction=meta["reaction_ids"][i],
                        minimum=float(lo.x[i]) if lo.success else None,maximum=float(hi.x[i]) if hi.success else None,
                        width=float(hi.x[i]-lo.x[i]) if lo.success and hi.success else None))
                row.update(face_objective_allowance=allowance,critical_ranges=ranges)
                net_ranges=[]
                shared_ids=sorted(meta["exchange_terms"])
                for name in args.net_metabolites:
                    if name not in shared_ids: continue
                    objective=kw["A_ub"].getrow(3+shared_ids.index(name)).toarray().ravel()
                    lo=linprog(objective,**face,method=method,options=options)
                    hi=linprog(-objective,**face,method=method,options=options)
                    net_ranges.append(dict(metabolite=name,cpu=float(objective@result.x),
                        minimum=float(objective@lo.x) if lo.success else None,
                        maximum=float(objective@hi.x) if hi.success else None,
                        width=float(objective@(hi.x-lo.x)) if lo.success and hi.success else None))
                row["net_uptake_ranges_mmol_l_h"]=net_ranges
        records.append(row)
        args.output.write_text(json.dumps(dict(scope="offline_diagnostic_not_training",lp=str(args.lp),records=records),indent=2))
        print(json.dumps(row),flush=True)


if __name__ == "__main__": main()
