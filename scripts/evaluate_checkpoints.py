"""
チェックポイント評価スクリプト
================================
学習時に VecNormalize の pkl が保存されなかったため、
全チェックポイントを新規 VecNormalize (training=False) で評価し、
相対的なランキングを求める。

Usage:
    # Phase 1: スクリーニング（100k ステップ間隔サンプリング）
    python scripts/evaluate_checkpoints.py --mode screening

    # Phase 2+3: 上位候補の詳細評価 + 時系列記録
    python scripts/evaluate_checkpoints.py --mode full

    # 特定エージェントのみ
    python scripts/evaluate_checkpoints.py --mode screening --agent godmode
    python scripts/evaluate_checkpoints.py --mode screening --agent pomdp
"""
import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.rl_environment_pomdp import RealWorldConsortiumEnv

# =====================================================================
# 定数
# =====================================================================
SBML_DIR = "models/sbml/final_consortium"
OUTPUT_DIR = Path("outputs/eval_results")
FIGURE_DIR = Path("paper_figures")

GODMODE_CKPT_DIR = Path("outputs/checkpoints")
POMDP_CKPT_DIR   = Path("outputs/checkpoints_pomdp")

# サンプリング間隔（スクリーニング用）
SCREENING_INTERVAL = 100_000
# 詳細評価エピソード数
DETAIL_N_EPISODES = 5
# 時系列記録用エピソード数
TIMESERIES_N_EPISODES = 1


# =====================================================================
# 環境ファクトリ
# =====================================================================
def load_models():
    all_models = load_sbml_models(Path(SBML_DIR))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    return models, initial_biomass, initial_metabolites


def make_godmode_env(models, initial_biomass, initial_metabolites):
    def _init():
        sim = dFBASimulator(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=100.0,
            volume=1.0,
            dt=0.2
        )
        return ConsortiumEnv(simulator=sim, max_time=168.0)
    return _init


def make_pomdp_env(models, initial_biomass, initial_metabolites):
    def _init():
        sim = dFBASimulator(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=100.0,
            volume=1.0,
            dt=0.2
        )
        return RealWorldConsortiumEnv(simulator=sim, max_time=168.0)
    return _init


# =====================================================================
# VecNormalize ラッパー構築
# =====================================================================
def build_eval_env(env_factory, vecnorm_pkl=None):
    """
    VecNormalize で包んだ評価用環境を返す。
    pkl が存在する場合はそれを読み込む（統計復元可能）。
    ない場合は新規作成 (training=False) でランキング比較に使う。
    """
    dummy = DummyVecEnv([env_factory])
    if vecnorm_pkl and os.path.exists(vecnorm_pkl):
        env = VecNormalize.load(vecnorm_pkl, dummy)
        env.training = False
        env.norm_reward = False
        print(f"  [VecNorm] Loaded stats from: {vecnorm_pkl}")
    else:
        env = VecNormalize(dummy, norm_obs=True, norm_reward=False, training=False)
        if vecnorm_pkl:
            print(f"  [VecNorm] pkl not found ({vecnorm_pkl}), using fresh stats.")
        else:
            print(f"  [VecNorm] No pkl path specified, using fresh stats.")
    return env


# =====================================================================
# 単一エピソード評価
# =====================================================================
def evaluate_one_episode(model, env, record_timeseries=False):
    """
    1エピソードを決定論的に実行し、最終 info と時系列データを返す。
    """
    obs = env.reset()
    done = False
    timeseries = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        if record_timeseries:
            info = infos[0]
            timeseries.append({
                'Time':               info.get('survival_hours', 0.0),
                'Total_PHA':          info.get('total_pha', 0.0),
                'Rubber_Remaining':   info.get('rubber_remaining', 0.0),
                'Rubber_Degraded':    info.get('total_rubber_degraded', 0.0),
                'pH':                 info.get('ph', 7.0),
                'DO':                 info.get('do', 0.25),
                'Biomass_OR16':       info.get('biomass_or16', 0.0),
                'Biomass_NS21':       info.get('biomass_ns21', 0.0),
                'Biomass_LP':         info.get('biomass_lp', 0.0),
            })

    final_info = infos[0]
    return final_info, timeseries


# =====================================================================
# チェックポイントのリスト取得（ステップ番号でソート）
# =====================================================================
def get_checkpoints(ckpt_dir, prefix):
    """
    指定ディレクトリから {prefix}_{step}_steps.zip を探し、
    (step, zip_path, pkl_path_or_None) のリストをステップ順で返す。
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    results = []
    for f in sorted(ckpt_dir.iterdir()):
        m = pattern.match(f.name)
        if m:
            step = int(m.group(1))
            zip_path = f
            # 対応する pkl を探す（save_vecnormalize=True で保存された場合）
            pkl_name = f"{prefix}_vecnormalize_{step}_steps.pkl"
            pkl_path = ckpt_dir / pkl_name
            results.append((step, zip_path, str(pkl_path) if pkl_path.exists() else None))
    results.sort(key=lambda x: x[0])
    return results


# =====================================================================
# Phase 1: スクリーニング
# =====================================================================
def run_screening(checkpoints, env_factory, agent_label, interval=SCREENING_INTERVAL):
    """
    interval ごとにサンプリングしたチェックポイントを評価。
    """
    # サンプリング: interval の倍数に最も近いものを選ぶ
    sampled = []
    seen_buckets = set()
    for step, zip_path, pkl_path in checkpoints:
        bucket = round(step / interval)
        if bucket not in seen_buckets:
            seen_buckets.add(bucket)
            sampled.append((step, zip_path, pkl_path))

    print(f"\n{'='*60}")
    print(f"[{agent_label}] Phase 1: Screening ({len(sampled)} checkpoints)")
    print(f"{'='*60}")

    rows = []
    for step, zip_path, pkl_path in sampled:
        print(f"  Evaluating step={step:>10,} ({zip_path.name})...")
        try:
            env = build_eval_env(env_factory, pkl_path)
            model = PPO.load(str(zip_path), env=env)
            final_info, _ = evaluate_one_episode(model, env)
            env.close()

            row = {
                'Agent':            agent_label,
                'Step':             step,
                'Checkpoint':       zip_path.name,
                'VecNorm_Restored': pkl_path is not None,
                'Total_PHA':        final_info.get('total_pha', 0.0),
                'Rubber_Degraded':  final_info.get('total_rubber_degraded', 0.0),
                'Survival_Hours':   final_info.get('survival_hours', 0.0),
                'Final_pH':         final_info.get('ph', 0.0),
                'Rubber_Remaining': final_info.get('rubber_remaining', 0.0),
            }
            rows.append(row)
            print(f"    → PHA={row['Total_PHA']:.2f} mmol, "
                  f"Degraded={row['Rubber_Degraded']:.2f} g/L, "
                  f"Survival={row['Survival_Hours']:.1f}h, "
                  f"pH={row['Final_pH']:.2f}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

    return pd.DataFrame(rows)


# =====================================================================
# Phase 2: 上位候補の詳細評価
# =====================================================================
def run_detail_evaluation(top_checkpoints, env_factory, agent_label, n_episodes=DETAIL_N_EPISODES):
    """
    top_checkpoints リスト（(step, zip_path, pkl_path)）を n_episodes ずつ評価。
    """
    print(f"\n{'='*60}")
    print(f"[{agent_label}] Phase 2: Detail Evaluation (n_ep={n_episodes})")
    print(f"{'='*60}")

    rows = []
    for step, zip_path, pkl_path in top_checkpoints:
        print(f"  Evaluating step={step:>10,} x{n_episodes} episodes...")
        ep_results = []
        for ep in range(n_episodes):
            try:
                env = build_eval_env(env_factory, pkl_path)
                model = PPO.load(str(zip_path), env=env)
                final_info, _ = evaluate_one_episode(model, env)
                env.close()
                ep_results.append({
                    'Total_PHA':       final_info.get('total_pha', 0.0),
                    'Rubber_Degraded': final_info.get('total_rubber_degraded', 0.0),
                    'Survival_Hours':  final_info.get('survival_hours', 0.0),
                    'Final_pH':        final_info.get('ph', 0.0),
                })
            except Exception as e:
                print(f"    [ERROR] ep={ep}: {e}")
                continue

        if ep_results:
            phas = [r['Total_PHA'] for r in ep_results]
            rows.append({
                'Agent':              agent_label,
                'Step':               step,
                'Checkpoint':         zip_path.name,
                'VecNorm_Restored':   pkl_path is not None,
                'N_Episodes':         len(ep_results),
                'PHA_Mean':           np.mean(phas),
                'PHA_Std':            np.std(phas),
                'PHA_Min':            np.min(phas),
                'PHA_Max':            np.max(phas),
                'Rubber_Degraded_Mean': np.mean([r['Rubber_Degraded'] for r in ep_results]),
                'Survival_Mean':      np.mean([r['Survival_Hours'] for r in ep_results]),
                'pH_Mean':            np.mean([r['Final_pH'] for r in ep_results]),
            })
            print(f"    → PHA_mean={rows[-1]['PHA_Mean']:.2f}±{rows[-1]['PHA_Std']:.2f} mmol")

    return pd.DataFrame(rows)


# =====================================================================
# Phase 3: 時系列記録
# =====================================================================
def run_timeseries(step, zip_path, pkl_path, env_factory, agent_label):
    """
    ベストモデルで168h シミュレーションを実行し、時系列DataFrameを返す。
    """
    print(f"\n{'='*60}")
    print(f"[{agent_label}] Phase 3: Timeseries Recording (step={step:,})")
    print(f"{'='*60}")

    env = build_eval_env(env_factory, pkl_path)
    model = PPO.load(str(zip_path), env=env)
    _, timeseries = evaluate_one_episode(model, env, record_timeseries=True)
    env.close()

    df = pd.DataFrame(timeseries)
    df['Agent_Type'] = agent_label
    print(f"  Recorded {len(df)} timesteps, final PHA={df['Total_PHA'].iloc[-1]:.2f} mmol")
    return df


# =====================================================================
# メイン
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Checkpoint Evaluation Script")
    parser.add_argument('--mode', choices=['screening', 'full'], default='screening')
    parser.add_argument('--agent', choices=['godmode', 'pomdp', 'both'], default='both')
    parser.add_argument('--top-n', type=int, default=3, help='Top N candidates for detail eval')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SBML models...")
    models, initial_biomass, initial_metabolites = load_models()

    godmode_factory = make_godmode_env(models, initial_biomass, initial_metabolites)
    pomdp_factory   = make_pomdp_env(models, initial_biomass, initial_metabolites)

    # チェックポイント一覧取得
    godmode_ckpts = get_checkpoints(GODMODE_CKPT_DIR, 'ppo_godmode_v3')
    pomdp_ckpts   = get_checkpoints(POMDP_CKPT_DIR,   'ppo_realworld')

    pkl_count_gm   = sum(1 for _, _, p in godmode_ckpts if p)
    pkl_count_pomdp = sum(1 for _, _, p in pomdp_ckpts if p)
    print(f"Godmode checkpoints: {len(godmode_ckpts)} (VecNorm pkl: {pkl_count_gm})")
    print(f"POMDP   checkpoints: {len(pomdp_ckpts)}   (VecNorm pkl: {pkl_count_pomdp})")

    # =====================================================================
    # Phase 1: スクリーニング
    # =====================================================================
    screening_dfs = []

    if args.agent in ('godmode', 'both'):
        df_gm = run_screening(godmode_ckpts, godmode_factory, 'God-Mode')
        screening_dfs.append(df_gm)

    if args.agent in ('pomdp', 'both'):
        df_pomdp = run_screening(pomdp_ckpts, pomdp_factory, 'Real-World (POMDP)')
        screening_dfs.append(df_pomdp)

    screening_df = pd.concat(screening_dfs, ignore_index=True)
    screening_path = OUTPUT_DIR / 'screening_results.csv'
    screening_df.to_csv(screening_path, index=False)
    print(f"\n[Saved] {screening_path}")

    # ランキング表示
    print("\n--- Screening Results (sorted by Total_PHA) ---")
    print(screening_df.sort_values('Total_PHA', ascending=False).to_string(index=False))

    if args.mode == 'screening':
        print("\nScreening complete. Run with --mode full for detail evaluation.")
        return

    # =====================================================================
    # Phase 2: 詳細評価
    # =====================================================================
    detail_dfs = []

    for agent_label, ckpts, factory in [
        ('God-Mode', godmode_ckpts, godmode_factory),
        ('Real-World (POMDP)', pomdp_ckpts, pomdp_factory),
    ]:
        if agent_label == 'God-Mode' and args.agent not in ('godmode', 'both'):
            continue
        if agent_label == 'Real-World (POMDP)' and args.agent not in ('pomdp', 'both'):
            continue

        # スクリーニング結果からtop-N抽出
        agent_screen = screening_df[screening_df['Agent'] == agent_label]
        top_steps = agent_screen.nlargest(args.top_n, 'Total_PHA')['Step'].tolist()
        top_ckpts = [(s, zp, pp) for s, zp, pp in ckpts if s in top_steps]

        print(f"\n[{agent_label}] Top-{args.top_n} candidates: steps = {top_steps}")
        df_detail = run_detail_evaluation(top_ckpts, factory, agent_label)
        detail_dfs.append(df_detail)

    detail_df = pd.concat(detail_dfs, ignore_index=True)
    detail_path = OUTPUT_DIR / 'top_candidates.csv'
    detail_df.to_csv(detail_path, index=False)
    print(f"\n[Saved] {detail_path}")

    print("\n--- Top Candidates (sorted by PHA_Mean) ---")
    print(detail_df.sort_values('PHA_Mean', ascending=False).to_string(index=False))

    # =====================================================================
    # Phase 3: 時系列記録（各エージェントのベストから1エピソード）
    # =====================================================================
    fig11_dfs = []

    for agent_label, ckpts, factory in [
        ('God-Mode', godmode_ckpts, godmode_factory),
        ('Real-World (POMDP)', pomdp_ckpts, pomdp_factory),
    ]:
        if agent_label == 'God-Mode' and args.agent not in ('godmode', 'both'):
            continue
        if agent_label == 'Real-World (POMDP)' and args.agent not in ('pomdp', 'both'):
            continue

        agent_detail = detail_df[detail_df['Agent'] == agent_label]
        if agent_detail.empty:
            print(f"[{agent_label}] No detail results, skipping timeseries.")
            continue

        best_step = agent_detail.loc[agent_detail['PHA_Mean'].idxmax(), 'Step']
        best_ckpt = [(s, zp, pp) for s, zp, pp in ckpts if s == best_step]
        if not best_ckpt:
            print(f"[{agent_label}] Best checkpoint (step={best_step}) not found.")
            continue

        step, zip_path, pkl_path = best_ckpt[0]
        ts_df = run_timeseries(step, zip_path, pkl_path, factory, agent_label)

        # 個別保存
        prefix = 'godmode' if 'God' in agent_label else 'pomdp'
        ts_path = OUTPUT_DIR / f'best_model_timeseries_{prefix}.csv'
        ts_df.to_csv(ts_path, index=False)
        print(f"[Saved] {ts_path}")

        fig11_dfs.append(ts_df)

    # Figure 11 用 CSV 生成
    if fig11_dfs:
        fig11_df = pd.concat(fig11_dfs, ignore_index=True)
        fig11_path = FIGURE_DIR / 'fig11_pomdp_comparison.csv'
        fig11_df.to_csv(fig11_path, index=False)
        print(f"\n[Saved] Figure 11 data: {fig11_path}")

    # ベストチェックポイント情報を JSON に保存
    best_info = {}
    for agent_label in detail_df['Agent'].unique():
        agent_detail = detail_df[detail_df['Agent'] == agent_label]
        best_row = agent_detail.loc[agent_detail['PHA_Mean'].idxmax()]
        best_info[agent_label] = {
            'step':       int(best_row['Step']),
            'checkpoint': best_row['Checkpoint'],
            'pha_mean':   float(best_row['PHA_Mean']),
            'pha_std':    float(best_row['PHA_Std']),
        }

    best_json_path = OUTPUT_DIR / 'best_checkpoints.json'
    with open(best_json_path, 'w') as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {best_json_path}")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    for agent, info in best_info.items():
        print(f"  [{agent}] Best: step={info['step']:,}, "
              f"PHA={info['pha_mean']:.2f}±{info['pha_std']:.2f} mmol")
    print(f"\nNext step (Phase 4): Update 'latest_checkpoint' in training scripts")
    print(f"to the best checkpoint path, then re-run training to resume.")


if __name__ == "__main__":
    main()
