"""Describe the completed development solver; not a speed comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output-stem", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.report.read_text())
    if len(source["runs"]) != 1: raise ValueError("One development rollout required")
    run = source["runs"][0]
    history = run["gpu"]["backend_history"]
    if not run["passed"] or run["gpu"]["steps"] != 120 or len(history) != 360:
        raise ValueError("Complete passing development rollout required")
    names = ["Minimum growth", "Aggregate objective", "Exchange parsimony"]
    data = []
    for i, name in enumerate(names):
        stages = history[i::3]
        data.append(dict(stage=name, lp_calls=len(stages),
            elapsed_seconds=sum(h["total_seconds"] for h in stages),
            host_presolve_seconds=sum(h["host_presolve_seconds"] for h in stages),
            phase_one_iterations=sum(h["gpu"]["phase_iterations"][0] for h in stages),
            phase_two_iterations=sum(h["gpu"]["phase_iterations"][1] for h in stages)))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.7,
        "svg.fonttype":"none", "savefig.dpi":300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), layout="constrained")
    positions = np.arange(3)
    axes[0].barh(positions, [d["elapsed_seconds"] for d in data], height=.58, color="#0072B2")
    axes[0].set(yticks=positions, yticklabels=names, xlabel="LP call time (s)")
    axes[0].invert_yaxis()
    phase1 = np.asarray([d["phase_one_iterations"] for d in data])
    phase2 = np.asarray([d["phase_two_iterations"] for d in data])
    percent = 100 * phase1 / (phase1 + phase2)
    axes[1].barh(positions, percent, height=.58, color="#0072B2", label="Phase I")
    axes[1].barh(positions, 100 - percent, left=percent, height=.58, color="#E69F00", label="Phase II")
    axes[1].set(yticks=positions, yticklabels=names, xlim=(0, 100),
                xlabel="Share of simplex iterations (%)")
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.18))
    for ax, panel in zip(axes, "ab"):
        ax.set_title(f"({panel})", loc="left", fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=".9", lw=.5)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"): fig.savefig(args.output_stem.with_suffix(suffix))
    plt.close(fig)
    args.output_stem.with_suffix(".json").write_text(json.dumps(dict(source=str(args.report),
        scope="One development rollout, seed 20260913; not independent qualification or controlled speed comparison",
        units="Panel a includes host and GPU work; panel b measures iterations, not elapsed time",
        stages=data), indent=2))


if __name__ == "__main__": main()
