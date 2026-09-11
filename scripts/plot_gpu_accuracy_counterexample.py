"""Plot a complete failed validation pair; never relabels it as qualified."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--state-comparison",type=Path,required=True)
    p.add_argument("--output-stem",type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.report.read_text()); r=report["runs"][0]
    cpu,gpu=r["exact"]["trajectory"],r["gpu"]["trajectory"]
    if len(cpu)!=120 or len(gpu)!=120 or not all(t["accepted"] for t in cpu+gpu):
        raise ValueError("Two complete accepted 120-step trajectories required")
    if r["passed"]: raise ValueError("This is a diagnostic figure for a failed test")
    comparison=json.loads(args.state_comparison.read_text())
    state16=next(item for item in comparison["records"] if item["step"]==16)
    pool_map={item["metabolite"]:item for item in state16["largest_pool_differences"]}
    names=["val__L_e","leu__L_e","for_e","h2_e"]
    deltas=[pool_map[n]["gpu"]-pool_map[n]["cpu"] for n in names]
    t=np.arange(1,121)*.2
    pha=np.asarray([100*abs(a["pha"]-b["pha"])/max(abs(a["pha"]),1e-9) for a,b in zip(cpu,gpu)])
    biomass=np.asarray([max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]) for a,b in zip(cpu,gpu)])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
        "svg.fonttype":"none","savefig.dpi":300})
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.9),layout="constrained")
    for ax,values,threshold,label in ((axes[0,0],pha,1.,"PHA error (%)"),
        (axes[0,1],biomass,.01,"Biomass error (g L$^{-1}$)")):
        ax.plot(t,np.maximum(values,1e-14),color="#0072B2",lw=1.)
        ax.axhline(threshold,color=".35",ls="--",lw=.8)
        ax.set(yscale="log",xlim=(0,24),xticks=[0,6,12,18,24],xlabel="Time (h)",ylabel=label)
        ax.grid(axis="y",color=".9",lw=.5)
    axes[1,0].plot(t,[a["nh4"] for a in cpu],color="#0072B2",lw=1.,label="CPU")
    axes[1,0].plot(t,[a["nh4"] for a in gpu],color="#D55E00",lw=1.,ls="--",label="GPU")
    axes[1,0].set(xlim=(21.5,24),ylim=(0,.45),xlabel="Time (h)",ylabel="NH$_4^+$ (mM)")
    axes[1,0].legend(frameon=False,ncol=2,loc="upper left")
    axes[1,1].bar(range(4),deltas,color=["#D55E00" if d>=0 else "#0072B2" for d in deltas],width=.6)
    axes[1,1].axhline(0,color=".4",lw=.7)
    axes[1,1].set(xticks=range(4),xticklabels=["Valine","Leucine","Formate","H$_2$"],
        ylabel="GPU − CPU at 3.2 h (mM)")
    axes[1,1].ticklabel_format(axis="y",style="sci",scilimits=(0,0),useMathText=True)
    for ax,panel in zip(axes.flat,"abcd"):
        ax.set_title(f"({panel})",loc="left",fontweight="bold")
    for suffix in (".png",".svg"): fig.savefig(args.output_stem.with_suffix(suffix))
    plt.close(fig)
    args.output_stem.with_suffix(".json").write_text(json.dumps(dict(seed=r["seed"],passed=False,
        source=str(args.report),state_source=str(args.state_comparison),endpoint_errors=r["errors"],
        pha_denominator_floor_g_l=1e-9,zero_plotting_floor=1e-14,
        pool_difference_step=16,pool_difference_time_h=3.2,
        pool_differences_mM=dict(zip(names,deltas))),indent=2))


if __name__=="__main__": main()
