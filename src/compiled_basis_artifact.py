"""Non-pickle storage for offline compiled LP bases (no executable payload)."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .gpu_certified_basis import NormalizedLP


def save_anchor(path,anchor):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    lp=anchor["lp"]
    data={"rhs":lp.rhs,"lower":lp.lower,"upper":lp.upper,"c":lp.c,"neq":lp.neq,
          "col_scale":lp.col_scale,"row_scale":lp.row_scale}
    for name,matrix in (("a",lp.a),("inverse",anchor["inverse"])):
        data.update({name+"_"+key:value for key,value in dict(data=matrix.data,indices=matrix.indices,
            indptr=matrix.indptr,shape=np.array(matrix.shape)).items()})
    for key in ("basic","active","kind","row_kind","cpu_anchor_values","cpu_anchor_objective",
                "compilation_seconds","offline_cpu_lp_calls","inverse_density"):
        data[key]=anchor[key]
    np.savez(path,**data)


def load_anchor(path):
    with np.load(path,allow_pickle=False) as data:
        def matrix(name):
            return csr_matrix((data[name+"_data"],data[name+"_indices"],data[name+"_indptr"]),shape=tuple(data[name+"_shape"]))
        lp=NormalizedLP(matrix("a"),data["rhs"],data["lower"],data["upper"],data["c"],int(data["neq"]),
            data["col_scale"],data["row_scale"])
        anchor=dict(lp=lp,inverse=matrix("inverse"))
        for key in ("basic","active","kind","row_kind","cpu_anchor_values"):
            anchor[key]=data[key]
        for key in ("cpu_anchor_objective","compilation_seconds","offline_cpu_lp_calls","inverse_density"):
            anchor[key]=data[key].item()
    n,m=lp.a.shape[1],lp.a.shape[0]
    if (lp.rhs.shape!=(m,) or lp.lower.shape!=(n,) or lp.upper.shape!=(n,) or lp.c.shape!=(n,) or
        anchor["inverse"].shape!=(len(anchor["basic"]),len(anchor["active"])) or
        len(anchor["basic"])!=len(anchor["active"]) or anchor["kind"].shape!=(n,) or anchor["row_kind"].shape!=(m,)):
        raise ValueError("Invalid compiled basis dimensions")
    return anchor
