#!/usr/bin/env python3
"""Plot completed frozen verification; no imputed or incomplete endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--qualification",type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.qualification.read_text())
    if len(manifest["runs"])!=5 or any("report" not in r for r in manifest["runs"]):
        raise ValueError("Five completed reports required; no partial endpoint substitution")
    reports=[]
    for row in manifest["runs"]:
        raw=(ROOT/row["report"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row.get("report_sha256"):
            raise ValueError("Report changed or is missing its qualification hash")
        reports.append(json.loads(raw))
    runs=[r["runs"][0] for r in reports]
    if any(r["gpu"]["steps"]!=120 or len(r["gpu"]["trajectory"])!=120 or
           not all(t["accepted"] for t in r["gpu"]["trajectory"]) for r in runs):
        raise ValueError("Incomplete trajectories: accuracy figure is not valid")
    if any(r["exact"]["steps"]!=120 or len(r["exact"]["trajectory"])!=120 or
           not all(t["accepted"] and "stage=parsimonious_exchange" in t.get("status","")
                   for side in ("exact","gpu") for t in r[side]["trajectory"]) for r in runs):
        raise ValueError("Incomplete three-stage CPU reference or GPU trajectory")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,
        "axes.titlesize":9,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.7,"xtick.major.width":.7,"ytick.major.width":.7,
        "svg.fonttype":"none","savefig.dpi":300})
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55),layout="constrained")
    fields=[("pha_relative",100.,1.,"PHA error (%)","#0072B2"),
            ("biomass_max",1.,.01,"Biomass error (g L$^{-1}$)","#D55E00"),
            ("phv_fraction",1.,.01,"3HV fraction error","#009E73")]
    summary=[]
    for ax,(field,scale,threshold,label,color),panel in zip(axes,fields,"abc"):
        values=np.asarray([r["errors"][field]*scale for r in runs])
        if not np.all(np.isfinite(values)) or np.any(values<0): raise ValueError("Invalid errors")
        # Any numerical zeros are explicitly recorded as below plotting floor.
        floor=1e-14
        ax.scatter(np.arange(1,6),np.maximum(values,floor),s=26,color=color,zorder=3)
        ax.axhline(threshold,color="0.35",ls="--",lw=.8)
        ax.set_yscale("log")
        ax.set_ylim(min(np.maximum(values,floor).min()/4,threshold/10),max(threshold*3,values.max()*3))
        ax.set(xlim=(.5,5.5),xticks=range(1,6),xlabel="Validation seed",ylabel=label)
        ax.set_title(f"({panel})",loc="left",fontweight="bold")
        ax.grid(axis="y",which="major",color=".9",lw=.5)
        summary.append(dict(metric=field,maximum=float(values.max()),threshold=threshold,
                            plotting_floor=floor,zeros_clipped=int(np.count_nonzero(values==0))))
    stem=args.qualification.parent/"accuracy_gates"
    fig.savefig(stem.with_suffix(".png")); fig.savefig(stem.with_suffix(".svg")); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),layout="constrained")
    trajectory_errors=[]
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00"]
    for seed_index,r in enumerate(runs):
        values=[]
        for cpu,gpu in zip(r["exact"]["trajectory"],r["gpu"]["trajectory"]):
            values.append([100*abs(cpu["pha"]-gpu["pha"])/max(abs(cpu["pha"]),1e-9),
                max(abs(cpu["biomass"][k]-gpu["biomass"][k]) for k in cpu["biomass"]),
                abs(cpu["phv_fraction"]-gpu["phv_fraction"])])
        values=np.asarray(values)
        trajectory_errors.append(dict(seed=r["seed"],max_pha_percent=float(values[:,0].max()),
            max_biomass_g_l=float(values[:,1].max()),max_phv_fraction=float(values[:,2].max())))
        for j,ax in enumerate(axes):
            ax.plot(np.arange(1,121)*.2,np.maximum(values[:,j],1e-14),lw=.8,
                    color=colors[seed_index],alpha=.85,label=str(seed_index+1))
    for ax,(_,_,threshold,label,_),panel in zip(axes,fields,"abc"):
        ax.axhline(threshold,color=".35",ls="--",lw=.8)
        ax.set(yscale="log",xlim=(0,24),xticks=[0,6,12,18,24],xlabel="Time (h)",ylabel=label)
        ax.set_title(f"({panel})",loc="left",fontweight="bold")
        ax.grid(axis="y",which="major",color=".9",lw=.5)
    axes[0].legend(title="Validation seed",ncol=5,loc="lower left",fontsize=6,
                   title_fontsize=6,frameon=False,columnspacing=.6,handlelength=1.)
    trajectory_stem=args.qualification.parent/"trajectory_errors"
    fig.savefig(trajectory_stem.with_suffix(".png")); fig.savefig(trajectory_stem.with_suffix(".svg")); plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.6,2.7),layout="constrained")
    for index,r in enumerate(runs):
        times=np.asarray([r["exact"]["seconds"],r["gpu"]["seconds"]])
        if not np.all(np.isfinite(times)) or np.any(times<=0): raise ValueError("Invalid run times")
        shift=(index-2)*.035
        ax.plot(np.asarray([0.,1.])+shift,times,"o-",color=colors[index],
                lw=.8,ms=4,label=str(index+1))
    ax.set(yscale="log",xlim=(-.25,1.25),xticks=[0,1],
           xticklabels=["CPU HiGHS","GPU simplex"],ylabel="Wall time (s)")
    ax.grid(axis="y",which="major",color=".9",lw=.5)
    ax.legend(title="Validation seed",frameon=False,fontsize=7,title_fontsize=7,
              loc="center left",bbox_to_anchor=(1.01,.5))
    runtime_stem=args.qualification.parent/"runtime_verification"
    fig.savefig(runtime_stem.with_suffix(".png")); fig.savefig(runtime_stem.with_suffix(".svg")); plt.close(fig)
    audit=[]
    for r in runs:
        history=r["gpu"]["backend_history"]
        audit.append(dict(seed=r["seed"],errors=r["errors"],
            max_original_residual=max(h["max_original_residual"] for h in history),
            cpu_lp_calls=r["gpu"]["cpu_lp_stage_calls"], gpu_lp_stages=r["gpu"]["gpu_lp_stage_calls"],
            gpu_lp_attempts=sum(h["gpu"].get("gpu_lp_attempts",1) for h in history),
            gpu_primal_repairs=sum(h["gpu"].get("gpu_primal_repairs",0) for h in history),
            max_cpu_optimizer_iterations=max(v for h in history for v in h["cpu_iteration_counts"].values()),
            gpu_seconds_verification=r["gpu"]["seconds"],cpu_seconds_verification=r["exact"]["seconds"],
            observed_time_ratio_gpu_over_cpu=r["gpu"]["seconds"]/r["exact"]["seconds"]))
    (args.qualification.parent/"figure_summary.json").write_text(json.dumps(dict(metrics=summary,runs=audit,
        trajectory_errors=trajectory_errors,trajectory_plotting_floor=1e-14,
        pha_denominator_floor_g_l=1e-9,
        runtime_note="One verification rollout per seed and backend; no timing confidence interval",
        qualification_timing_note=manifest["timing_note"]),indent=2))
    print(str(stem.with_suffix(".png")))


if __name__=="__main__": main()
