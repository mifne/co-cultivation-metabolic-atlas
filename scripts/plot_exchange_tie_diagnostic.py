"""Early-trajectory diagnostic, explicitly not a full-horizon qualification."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    sources = {
        "CPU reference": RESULTS/"pf_seed20286001_cpu_diagnostic/replay.json",
        "GPU, 3 LPs": RESULTS/"pf_seed20286001_gpu_diagnostic17/replay.json",
        "GPU, 4 LPs": RESULTS/"pf_seed20286001_gpu_exchange_tie17/replay.json",
        "CPU, IPM": RESULTS/"pf_seed20286001_cpu_ipm_sensitivity120.json",
    }
    rows = {name: json.loads(path.read_text())["rows"][:17] for name, path in sources.items()}
    if any(len(trace) != 17 or [r["step"] for r in trace] != list(range(1, 18)) for trace in rows.values()):
        raise ValueError("Complete aligned 17-step data required")
    cpu = rows.pop("CPU reference")
    palette = ["#0072B2", "#D55E00", "#009E73"]
    styles = ["-", "--", ":"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.linewidth": .7, "svg.fonttype": "none",
        "pdf.fonttype": 42, "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), layout="constrained")
    values = {}
    for (label, trace), color, style in zip(rows.items(), palette, styles):
        biomass = []
        pools = []
        for a, b in zip(cpu, trace):
            a, b = a["after"], b["after"]
            biomass.append(max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]))
            keys = set(a["metabolites"]) | set(b["metabolites"])
            pools.append(max(abs(a["metabolites"].get(k, 0)-b["metabolites"].get(k, 0)) for k in keys))
        values[label] = dict(biomass_difference_g_l=biomass, max_pool_difference_mM=pools)
        axes[0].plot(np.arange(1, 18)*.2, np.maximum(biomass, 1e-13), color=color, ls=style, lw=1.6, label=label)
        axes[1].plot(np.arange(1, 18)*.2, np.maximum(pools, 1e-13), color=color, ls=style, lw=1.6)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_xlabel("Time (h)")
        ax.set_xlim(0, 3.4)
        ax.grid(axis="y", color=".9", linewidth=.6)
    axes[0].set_ylabel("Max. biomass difference (g L$^{-1}$)")
    axes[1].set_ylabel("Max. broth concentration difference (mM)")
    names = ["val__L_e", "leu__L_e", "nh4_e", "h2_e"]
    x = np.arange(len(names))
    for j, ((label, trace), color) in enumerate(zip(rows.items(), palette)):
        delta = [trace[15]["after"]["metabolites"].get(k, 0)-cpu[15]["after"]["metabolites"].get(k, 0) for k in names]
        axes[2].bar(x+(j-1)*.24, np.asarray(delta)*1000, width=.23, color=color)
        values[label]["step16_delta_mM"] = dict(zip(names, delta))
    axes[2].set_xticks(x, ["Valine", "Leucine", "NH$_4^+$", "H$_2$"])
    axes[2].set_ylabel("Difference at 3.2 h (µM)")
    axes[2].axhline(0, color=".4", lw=.7)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    for i, ax in enumerate(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, 1.035, f"({chr(97+i)})", transform=ax.transAxes, weight="bold", fontsize=12)
    stem = RESULTS/"pf_exchange_tie_early_diagnostic"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(dict(
        scope="first 17 steps of previously failed seed 20286001; not full-horizon accuracy qualification",
        reference="original CPU HiGHS dual simplex", log_display_floor=1e-13,
        sources={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k, p in sources.items()},
        values=values), indent=2))


if __name__ == "__main__":
    main()
