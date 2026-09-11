"""Before/after numerical errors for an already-used regression condition."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"results"


def main():
    old_path = RESULTS/"pf_gpu_simplex_qualification_repaired_20260903/seed_20286001.json"
    new_path = RESULTS/"pf_seed20286001_gpu_exchange_tie120_audit.json"
    old = json.loads(old_path.read_text())["runs"][0]
    new = json.loads(new_path.read_text())
    if not new["passed"] or len(new["trajectory_errors"]) != 120:
        raise ValueError("Complete audited regression required")
    before = []
    for a, b in zip(old["exact"]["trajectory"], old["gpu"]["trajectory"]):
        before.append(dict(pha_relative=abs(a["pha"]-b["pha"])/max(abs(a["pha"]), 1e-9),
            biomass_g_l=max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]),
            phv_fraction=abs(a["phv_fraction"]-b["phv_fraction"])))
    if len(before) != 120:
        raise ValueError("Complete old trajectory required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), layout="constrained")
    specs = [("pha_relative", 100., "PHA error (%)", 1.),
        ("biomass_g_l", 1., "Max. biomass error (g L$^{-1}$)", .01),
        ("phv_fraction", 1., "3HV mole-fraction error", .01)]
    time = np.arange(1, 121)*.2
    for i, (ax, (key, scale, label, gate)) in enumerate(zip(axes, specs)):
        ax.plot(time, np.maximum([r[key]*scale for r in before], 1e-13), color="#0072B2", lw=1.5, label="GPU, 3 LPs")
        ax.plot(time, np.maximum([r[key]*scale for r in new["trajectory_errors"]], 1e-13),
            color="#D55E00", lw=1.5, ls="--", label="GPU, 4 LPs")
        ax.axhline(gate, color=".35", ls=":", lw=1., label="Endpoint criterion")
        ax.set(xlim=(0, 24), yscale="log", xlabel="Time (h)", ylabel=label)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=.6)
        ax.text(0, 1.035, f"({chr(97+i)})", weight="bold", fontsize=12, transform=ax.transAxes)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    stem = RESULTS/"pf_exchange_tie_full_regression"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(dict(
        scope="same previously failed seed, regression not independent qualification",
        reference="unchanged CPU HiGHS dual simplex, 3 stages", seed=20286001,
        display_floor_in_each_axis_units=1e-13,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, new_path)},
        old_endpoint=before[-1], new_endpoint=new["endpoint_errors"]), indent=2))


if __name__ == "__main__":
    main()
