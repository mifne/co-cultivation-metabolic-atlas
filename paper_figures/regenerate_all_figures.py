"""
Master Pipeline: Paper Figure Regeneration
Uses the latest God-mode refined model (godmode_final_refined.zip)
Execution order:
  Phase 1 - Data Extraction (simulations)
  Phase 2 - Plotting (publication-ready figures)
"""

import os
import subprocess
import sys

def run_cmd(cmd_args, desc, abort_on_fail=True):
    print(f"\n{'='*65}")
    print(f"[RUN] {desc}")
    print(f"Cmd:  {' '.join(cmd_args)}")
    print(f"{'='*65}")

    env = os.environ.copy()
    lib_path = "/home/reiya/miniforge3/envs/co-cultivation/lib"
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        res = subprocess.run(cmd_args, env=env, check=True, text=True,
                             capture_output=False)
        print(f"[OK] {desc}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {desc}  (exit code {e.returncode})")
        if abort_on_fail:
            sys.exit(1)
        return False


def py(script_path):
    return ["/home/reiya/miniforge3/envs/co-cultivation/bin/python", script_path]


def main():
    print("=" * 65)
    print(" Master Paper Figures Regeneration Pipeline")
    print(" Model: outputs/checkpoints/ppo_godmode_v3_550000_steps.zip")
    print("=" * 65)

    # ── Phase 1: Data Extraction ──────────────────────────────────
    # (order matters – fig3B data is also used by fig6)
    extractions = [
        (py("paper_figures/extract_fig2_data.py"),    "Fig 2 – Learning curve + time-course + heatmaps", True),
        (py("paper_figures/extract_fig3_data.py"),    "Fig 3 – RL vs Dual-PI vs Constant (1-min dt, 168h)", True),
        (py("paper_figures/extract_fig4_5_data.py"),  "Fig 4/5 – Training evolution + product optimisation", True),
        (py("paper_figures/extract_fig5_hybrid.py"),  "Fig 5 – Hybrid (deterministic + 2-stage) run", True),
        (py("paper_figures/extract_fig5_stochastic.py"), "Fig 5 – Stochastic best-of-3 rollout", False),
        (py("paper_figures/extract_fig7_data.py"),    "Fig 7 – pH acid shock recovery (168h)", True),
        (py("paper_figures/extract_fig8_xai.py"),     "Fig 8/10 – XAI feature importance + pacing", True),
        (py("paper_figures/extract_fig9_flux.py"),    "Fig 9 – Metabolic flux log", True),
        (py("scripts/extract_final_fig11.py"),        "Fig 11 – Final God-mode vs POMDP clean run", True),
        (py("scripts/run_information_deprivation.py"), "Fig 11 – Information deprivation experiment", True),
    ]

    # ── Phase 2: Plotting ─────────────────────────────────────────
    plots = [
        (py("paper_figures/plot_fig2.py"),        "Plot Figure 2 – Base control overview"),
        (py("paper_figures/plot_fig3.py"),        "Plot Figure 3 – RL vs Baselines"),
        (py("paper_figures/plot_fig4.py"),        "Plot Figure 4 – Survival & population dynamics"),
        (py("paper_figures/plot_fig5_revised.py"), "Plot Figure 5 – Product optimisation"),
        (py("paper_figures/plot_fig6_7.py"),      "Plot Figure 6 & 7 – Resource efficiency + shock recovery"),
        (py("paper_figures/plot_fig8_10.py"),     "Plot Figure 8 & 10 – XAI + pacing"),
        (py("paper_figures/plot_fig9.py"),        "Plot Figure 9 – Metabolic flux handoff"),
        (py("paper_figures/plot_fig11_final.py"), "Plot Figure 11 – Final comparison"),
    ]

    print("\n>>> Phase 1: Data Extraction <<<")
    for cmd, desc, abort in extractions:
        run_cmd(cmd, desc, abort_on_fail=abort)

    print("\n>>> Phase 2: Plotting <<<")
    for cmd, desc in plots:
        run_cmd(cmd, desc, abort_on_fail=False)

    print("\n" + "=" * 65)
    print("🎉  Pipeline complete!  All figures saved to paper_figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()