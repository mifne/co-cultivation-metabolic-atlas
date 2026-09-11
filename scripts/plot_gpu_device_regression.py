"""Publication-style numerical error curves; development regression, not qualification."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/"results/pf_seed20286001_gpu_exchange_tie120_audit.json",
             ROOT/"results/pf_gpu_device_lu_regression120_20260903/audit.json"]
    reports = [json.loads(path.read_text()) for path in paths]
    if any(not r["passed"] or len(r["trajectory_errors"]) != 120 or r["seed"] != 20286001 for r in reports):
        raise ValueError("Two complete passing regressions of the same seed required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), layout="constrained")
    specs = [("pha_relative", 100., "PHA error (%)", 1.),
        ("biomass_g_l", 1., "Max. biomass error (g L$^{-1}$)", .01),
        ("phv_fraction", 1., "3HV mole-fraction error", .01)]
    time = np.arange(1, 121)*.2
    for i, (ax, (key, scale, label, gate)) in enumerate(zip(axes, specs)):
        for report, color, style, name in zip(reports, ["#0072B2", "#D55E00"], ["-", "--"],
                ["Reference GPU", "Device-pivot GPU"]):
            ax.plot(time, np.maximum([r[key]*scale for r in report["trajectory_errors"]], 1e-13),
                    color=color, ls=style, lw=1.4, label=name)
        ax.axhline(gate, color=".35", ls=":", lw=1., label="Endpoint criterion")
        ax.set(xlim=(0, 24), yscale="log", xlabel="Time (h)", ylabel=label)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=.6)
        ax.text(0, 1.035, f"({chr(97+i)})", weight="bold", fontsize=12, transform=ax.transAxes)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    stem = ROOT/"results/pf_gpu_device_regression_errors_20260903"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(dict(seed=20286001,
        scope="development regression; both GPU variants vs the unchanged CPU reference",
        caption="Absolute/relative errors along one 24-hour development trajectory; dotted lines are endpoint gates. No independent-five-seed or speedup claim.",
        display_floor=1e-13, source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}), indent=2))


if __name__ == "__main__":
    main()
