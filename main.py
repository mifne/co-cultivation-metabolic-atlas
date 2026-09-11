"""
Main Script for dFBA-RL Consortium Control
天然ゴム分解コンソーシアムのdFBA-RL制御メインスクリプト
"""

import argparse
import numpy as np
from pathlib import Path
import json
import signal
import sys
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import cobra
from typing import Dict, Tuple, Optional, List, Callable

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from src.gpu_assignment import assign_gpu_for_worker
from src.fba_surrogate_service import start_surrogate_service
from src.callbacks import ConsortiumCallback
from src.utils import load_sbml_models, select_consortium_models, get_initial_params, create_mock_models


def load_requested_models(sbml_dir: str | None, profile: str = 'legacy3') -> dict:
    """Read an explicit consortium profile without creating or curating GEMs."""
    project = Path(__file__).resolve().parent
    models = load_sbml_models(Path(sbml_dir) if sbml_dir else project / 'models/sbml/final_consortium')
    if profile == 'pf-helper3' and not any('freudenreichii' in name.lower() for name in models):
        helper_path = project / 'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml'
        if not helper_path.is_file():
            raise FileNotFoundError(f'Requested Pf GEM is missing: {helper_path}')
        models['Propionibacterium_freudenreichii_shermanii'] = cobra.io.read_sbml_model(str(helper_path))
    return select_consortium_models(models, profile=profile)


def cultivation_options(args) -> dict:
    dynamics = getattr(args, 'dynamics', 'legacy')
    mode = getattr(args, 'fba_mode', None) or ('separate' if dynamics in {'audited','physiology'} else 'cooperative')
    backend = getattr(args, 'solver_backend', 'highs')
    if dynamics in {'audited','physiology'} and (mode != 'separate' or backend != 'highs'):
        raise ValueError('Audited dynamics require --fba-mode separate and --solver-backend highs')
    physiology = {}
    if getattr(args, 'physiology_config', None):
        if dynamics != 'physiology':
            raise ValueError('--physiology-config requires --dynamics physiology')
        physiology = json.loads(Path(args.physiology_config).read_text(encoding='utf-8'))
        if not isinstance(physiology, dict) or set(physiology)-{'maintenance','basal_death_rates','starvation_death_rates','nitrogen_policy','remobilize_pha'}:
            raise ValueError('unsupported physiology configuration')
    control_dt = float(getattr(args, 'control_dt', .2))
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError('control-dt must be positive and finite')
    return dict(
        dynamics=dynamics, consortium_profile=getattr(args, 'consortium_profile', 'legacy3'), physiology=physiology,
        control_dt=control_dt, internal_dt=getattr(args, 'internal_dt', .025),
        fba_mode=mode, solver_backend=backend,
        initial_rubber=getattr(args, 'initial_rubber', 100.),
        initial_nh4=getattr(args, 'initial_nh4', None),
        ph_control_target=getattr(args, 'ph_target', None) if getattr(args, 'ph_target', None) is not None
                          else (7. if dynamics in {'audited','physiology'} else None),
        observation_schema=getattr(args, 'observation_schema', None) or ('metabolic_v2' if dynamics in {'audited','physiology'} else 'legacy_v1'),
        max_specific_feed_rate_mmol_l_h=getattr(args, 'max_specific_feed_rate', .1),
    )


def make_env(
    sbml_dir: str,
    env_params: dict,
    data_log_path: Optional[str] = None,
    rank: int = 0,
    preloaded_models: Optional[dict] = None,
):
    """
    環境作成用のファクトリ関数を返す。
    """
    def _init():
        assigned_gpu = assign_gpu_for_worker(
            rank,
            env_params.get('gpu_ids'),
            env_params.get('gpu_slots_per_device', 1),
        )
        # Linux ``fork`` workers inherit these read-mostly GEM objects with
        # copy-on-write, avoiding a complete SBML copy per environment.
        if preloaded_models is not None:
            models = preloaded_models
        else:
            models = load_requested_models(sbml_dir, env_params.get('consortium_profile', 'legacy3'))
            
        initial_biomass, initial_metabolites = get_initial_params(models)
        if env_params.get('initial_nh4') is not None:
            initial_metabolites['nh4_e'] = float(env_params['initial_nh4'])
        
        # 並列環境ごとに異なるログファイル名を生成
        env_log_path = None
        if data_log_path:
            log_path_obj = Path(data_log_path)
            env_log_path = str(log_path_obj.parent / f"{log_path_obj.stem}_env{rank}{log_path_obj.suffix}")

        surrogate_service = env_params.get('surrogate_service')
        surrogate_services = env_params.get('surrogate_services') or []
        if surrogate_services:
            surrogate_service = surrogate_services[rank % len(surrogate_services)]

        simulator_type = dFBASimulator
        extra_simulator_kwargs = {}
        if env_params.get('dynamics', 'legacy') in {'audited','physiology'}:
            from src.audited_dfba import AuditedDFBASimulator
            if env_params.get('fba_mode', 'separate') != 'separate' or env_params.get('solver_backend', 'highs') != 'highs':
                raise ValueError('Audited dynamics require separate FBA and highs')
            simulator_type = AuditedDFBASimulator
            if env_params['dynamics'] == 'physiology':
                from src.physiology_dfba import PhysiologyDFBASimulator
                simulator_type = PhysiologyDFBASimulator
                extra_simulator_kwargs.update(env_params.get('physiology', {}))
            extra_simulator_kwargs['max_internal_dt'] = env_params.get('internal_dt', .025)
        sim = simulator_type(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=env_params.get('initial_rubber', 100.0),
            volume=1.0,
            dt=env_params.get('control_dt', .2),
            data_log_path=env_log_path,
            solver_backend=env_params.get('solver_backend', 'highs'),
            cuopt_method=env_params.get('cuopt_method', 'pdlp'),
            fba_mode=env_params.get('fba_mode', 'separate' if simulator_type is not dFBASimulator else 'cooperative'),
            ph_control_target=env_params.get('ph_control_target'),
            surrogate_dir=env_params.get('surrogate_dir'),
            surrogate_device=env_params.get('surrogate_device', 'cuda'),
            surrogate_service=surrogate_service,
            surrogate_audit_interval=env_params.get('surrogate_audit_interval', 128),
            surrogate_objective_rtol=env_params.get('surrogate_objective_rtol', 0.02),
            surrogate_ood_threshold=env_params.get('surrogate_ood_threshold', 8.0),
            cooperative_parsimony=env_params.get('cooperative_parsimony', True),
            cooperative_highs_presolve=env_params.get('cooperative_highs_presolve', True),
            cooperative_highs_method=env_params.get('cooperative_highs_method', 'highs'),
            cooperative_capture_training_snapshot=env_params.get(
                'cooperative_capture_training_snapshot', False
            ),
            cooperative_surrogate_artifact=env_params.get(
                'cooperative_surrogate_artifact'
            ),
            cooperative_surrogate_device=env_params.get(
                'cooperative_surrogate_device', 'cuda'
            ),
            cooperative_surrogate_top_k=env_params.get(
                'cooperative_surrogate_top_k', 16
            ),
            cooperative_surrogate_exact_interval=env_params.get(
                'cooperative_surrogate_exact_interval', 128
            ),
            cooperative_surrogate_require_qualified=env_params.get(
                'cooperative_surrogate_require_qualified', True
            ),
            cooperative_surrogate_validation_manifest=env_params.get(
                'cooperative_surrogate_validation_manifest'
            ),
            cooperative_surrogate_distance_threshold=env_params.get(
                'cooperative_surrogate_distance_threshold'
            ),
            cooperative_gpu_qp_projection=env_params.get(
                'cooperative_gpu_qp_projection', False
            ),
            cooperative_gpu_qp_only=env_params.get(
                'cooperative_gpu_qp_only', False
            ),
            cooperative_gpu_qp_candidates=env_params.get(
                'cooperative_gpu_qp_candidates', 128
            ),
            cooperative_gpu_qp_service=env_params.get(
                'cooperative_gpu_qp_service'
            ),
            **extra_simulator_kwargs,
        )
        if assigned_gpu is not None:
            sim.assigned_gpu = assigned_gpu
        
        env_kwargs = {
            key: value for key, value in env_params.items()
            if key not in {
                'solver_backend', 'cuopt_method', 'fba_mode', 'gpu_ids',
                'gpu_slots_per_device', 'surrogate_dir', 'surrogate_device',
                'surrogate_service', 'surrogate_audit_interval',
                'surrogate_services',
                'surrogate_objective_rtol', 'surrogate_ood_threshold',
                'cooperative_parsimony', 'cooperative_highs_presolve',
                'cooperative_highs_method',
                'cooperative_capture_training_snapshot',
                'cooperative_surrogate_artifact', 'cooperative_surrogate_device',
                'cooperative_surrogate_top_k',
                'cooperative_surrogate_exact_interval',
                'cooperative_surrogate_require_qualified',
                'cooperative_surrogate_validation_manifest',
                'cooperative_surrogate_distance_threshold',
                'cooperative_gpu_qp_projection',
                'cooperative_gpu_qp_only',
                'cooperative_gpu_qp_candidates',
                'cooperative_gpu_qp_service',
                'dynamics', 'consortium_profile', 'control_dt', 'internal_dt', 'physiology',
                'initial_rubber', 'initial_nh4', 'ph_control_target',
            }
        }
        env = ConsortiumEnv(simulator=sim, **env_kwargs)
        return env
    return _init










def setup_simulator(models: dict, data_log_path: Optional[str] = None) -> dFBASimulator:
    """
    dFBAシミュレーターをセットアップ
    """
    initial_biomass, initial_metabolites = get_initial_params(models)
    initial_rubber = 100.0
    dt = 0.2
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=initial_rubber,
        volume=1.0,
        dt=dt,
        data_log_path=data_log_path
    )
    return simulator


def convert_to_serializable(obj):
    """
    JSONシリアル化不可能なオブジェクトを変換
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    線形学習率スケジュール
    :param initial_value: 初期の学習率
    :return: 残りのステップ数（1.0から0.0）に応じた学習率を返す関数
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train_agent(args):
    """エージェントを訓練"""
    print("🧬 訓練開始")
    
    # モデルの読み込み
    options = cultivation_options(args)
    models = load_requested_models(getattr(args, 'sbml_dir', None), options['consortium_profile'])
    
    # One parameter source is shared by training and evaluation factories.
    env_params = {
        'max_time': args.max_steps * options['control_dt'],
        'solver_backend': getattr(args, 'solver_backend', 'highs'),
        'cuopt_method': getattr(args, 'cuopt_method', 'pdlp'),
        'fba_mode': getattr(args, 'fba_mode', 'cooperative'),
        'gpu_ids': getattr(args, 'gpu_ids', None),
        'gpu_slots_per_device': getattr(args, 'gpu_slots_per_device', 1),
        'surrogate_dir': getattr(args, 'surrogate_dir', None),
        'surrogate_device': getattr(args, 'surrogate_device', 'cuda'),
        'surrogate_audit_interval': getattr(args, 'surrogate_audit_interval', 128),
        'surrogate_objective_rtol': getattr(args, 'surrogate_objective_rtol', 0.02),
        'surrogate_ood_threshold': getattr(args, 'surrogate_ood_threshold', 8.0),
    }
    env_params.update(options)

    surrogate_managers = []
    if env_params['solver_backend'] == 'surrogate' and args.n_envs > 1:
        if not env_params['surrogate_dir']:
            raise ValueError('--surrogate-dir is required for the surrogate backend')
        raw_gpu_ids = env_params.get('gpu_ids') or os.environ.get('DFBA_GPU_IDS')
        service_devices = []
        if raw_gpu_ids:
            service_devices = [
                f"cuda:{int(value.strip())}"
                for value in str(raw_gpu_ids).split(',')
                if value.strip()
            ]
        if not service_devices:
            service_devices = [env_params['surrogate_device']]
        surrogate_services = []
        for service_device in service_devices:
            manager, service = start_surrogate_service(
                models,
                env_params['surrogate_dir'],
                device=service_device,
                batch_window_ms=getattr(args, 'surrogate_batch_window_ms', 2.0),
                max_batch_size=getattr(args, 'surrogate_max_batch_size', 64),
                ood_threshold=env_params['surrogate_ood_threshold'],
            )
            surrogate_managers.append(manager)
            surrogate_services.append(service)
        env_params['surrogate_services'] = surrogate_services
        # Each manager owns one CUDA context; workers are round-robin sharded.
        import atexit
        for manager in surrogate_managers:
            atexit.register(manager.shutdown)

    # 学習率スケジュールの設定
    lr = args.learning_rate
    if getattr(args, 'linear_lr', False):
        print(f"📉 線形学習率スケジュールを適用 (Initial LR: {lr})")
        lr = linear_schedule(lr)

    # The surrogate service owns CUDA. Environment workers only update their
    # private copy-on-write GEM bounds, so fork can share the large immutable
    # portion of the three models safely and substantially reduce host RAM.
    subproc_start_method = None
    shared_worker_models = None
    if (
        args.n_envs > 1
        and env_params["solver_backend"] == "surrogate"
        and "fork" in mp.get_all_start_methods()
    ):
        subproc_start_method = "fork"
        shared_worker_models = models

    # 環境の作成
    if args.n_envs > 1:
        # SubprocVecEnv用の関数のリストを作成
        env_input = [
            make_env(
                args.sbml_dir,
                env_params,
                getattr(args, "data_log_path", None),
                rank=i,
                preloaded_models=shared_worker_models,
            )
            for i in range(args.n_envs)
        ]
    else:
        # 単一環境（DummyVecEnv用）
        env_input = make_env(args.sbml_dir, env_params, getattr(args, 'data_log_path', None), rank=0)

    # PPOエージェントの作成
    agent = ConsortiumPPOAgent(
        env=env_input,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        device=getattr(args, 'device', 'cpu'),
        verbose=1,
        n_envs=args.n_envs,
        tensorboard_log=args.tensorboard_log,
        subproc_start_method=subproc_start_method,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = output_dir / 'cultivation_configuration.json'
    configuration = dict(cli_options=options,
                         environments=agent.env.env_method('get_cultivation_configuration'))
    if configuration_path.exists():
        if json.loads(configuration_path.read_text(encoding='utf-8')) != configuration:
            raise ValueError('Existing output cultivation configuration differs; choose a new output directory')
    else:
        with configuration_path.open('x', encoding='utf-8') as config_file:
            json.dump(configuration, config_file, indent=2, allow_nan=False)

    # チェックポイントから再開
    if args.resume_from:
        print(f"🔄 チェックポイントから再開: {args.resume_from}")
        agent.load(args.resume_from)

    # 訓練
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    callbacks = [
        ConsortiumCallback(),
        CheckpointCallback(
            save_freq=max(1000, args.save_freq // args.n_envs),
            save_path=args.output_dir,
            name_prefix='ppo_consortium',
            save_vecnormalize=True
        )
    ]
    
    print(f"🚀 学習開始: {args.total_timesteps} ステップ")
    print(f"📡 リアルタイム監視: TensorBoard (Science/ セクション)")
    print(f"💾 チェックポイント保存: {args.output_dir}")

    history = agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        callback=callbacks
    )
    
    # モデルの保存
    model_path = output_dir / 'ppo_consortium_model.zip'
    agent.save(str(model_path))
    
    # 訓練履歴の保存
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        serializable_history = convert_to_serializable(history)
        json.dump(serializable_history, f, indent=2)
    print(f"📊 訓練履歴保存: {history_path}")
    
    # Evaluate the trained environment with frozen normalization statistics.
    print("📊 最終評価中...")
    results = agent.evaluate(n_episodes=args.eval_episodes)
    
    # 結果の保存
    serializable_results = convert_to_serializable(results)
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ 訓練完了: {results_path}")


def evaluate_agent(args):
    """訓練済みエージェントを評価"""
    print("📊 評価開始")
    
    options = cultivation_options(args)
    options.update(max_time=args.max_steps * options['control_dt'],
                   cuopt_method=getattr(args, 'cuopt_method', 'pdlp'),
                   surrogate_dir=getattr(args, 'surrogate_dir', None),
                   surrogate_device=getattr(args, 'surrogate_device', 'cuda'),
                   surrogate_audit_interval=getattr(args, 'surrogate_audit_interval', 1),
                   surrogate_objective_rtol=getattr(args, 'surrogate_objective_rtol', .02),
                   surrogate_ood_threshold=getattr(args, 'surrogate_ood_threshold', 8.))
    env = make_env(args.sbml_dir, options)
    
    # エージェントの読み込み
    agent = ConsortiumPPOAgent(env=env)
    agent.load(args.model_path)
    
    # 評価
    results = agent.evaluate(n_episodes=args.eval_episodes)
    
    print("✅ 評価完了")


def main():
    parser = argparse.ArgumentParser(
        description='天然ゴム分解コンソーシアムのdFBA-RL制御システム'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='実行モード')
    
    # 訓練モード
    train_parser = subparsers.add_parser('train', help='エージェントを訓練')
    train_parser.add_argument('--sbml-dir', type=str, help='SBMLファイルのディレクトリ')
    train_parser.add_argument('--resume-from', type=str, default=None,
                             help='再開するモデルのパス')
    train_parser.add_argument('--total-timesteps', type=int, default=100000,
                             help='総訓練ステップ数')
    train_parser.add_argument('--max-steps', type=int, default=840,
                             help='1エピソードの最大ステップ数 (dt=0.2なら840で168h)')
    train_parser.add_argument('--target-degradation', type=float, default=0.9,
                             help='目標ゴム分解率')
    train_parser.add_argument('--amino-acid-cost', type=float, default=0.1,
                             help='アミノ酸コスト係数')
    train_parser.add_argument('--nutrient-cost', type=float, default=0.01,
                             help='基本栄養素（グルコース等）コスト係数')
    train_parser.add_argument('--data-log-path', type=str, default=None,
                             help='FBA結果のログ保存パス (サロゲートモデル用)')
    train_parser.add_argument('--learning-rate', type=float, default=5e-5,
                             help='学習率')
    train_parser.add_argument('--n-steps', type=int, default=4096,
                             help='PPOのn_steps (各環境での収集ステップ数)')
    train_parser.add_argument('--batch-size', type=int, default=512,
                             help='PPOのbatch_size')
    train_parser.add_argument('--ent-coef', type=float, default=0.01,
                             help='PPOのentropy係数')
    train_parser.add_argument('--gamma', type=float, default=0.999,
                             help='割引率')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='出力ディレクトリ')
    train_parser.add_argument('--log-interval', type=int, default=1,
                             help='ログ出力間隔 (PPO更新ごと)')
    train_parser.add_argument('--save-freq', type=int, default=5000,
                             help='チェックポイント保存頻度')
    train_parser.add_argument('--eval-episodes', type=int, default=5,
                             help='評価エピソード数')
    train_parser.add_argument('--eval-during-training', action='store_true',
                             help='訓練中に定期的に評価を実行')
    train_parser.add_argument('--eval-freq', type=int, default=5000,
                             help='訓練中評価の頻度')
    train_parser.add_argument('--n-envs', type=int, default=4,
                              help='並列環境数')
    train_parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu',
                              help='PPO方策の実行デバイス（環境シミュレーションは別途ソルバー設定）')
    train_parser.add_argument('--tensorboard-log', type=str, default='outputs/tensorboard',
                             help='TensorBoardのログ保存先')
    train_parser.add_argument('--linear-lr', action='store_true',
                             help='学習率の線形減衰を有効にする')
    train_parser.add_argument('--solver-backend', choices=['glpk', 'highs', 'cuopt', 'auto', 'surrogate'], default='highs',
                             help='FBAソルバー。cuOptはNVIDIA GPU用、autoは未導入時にGLPKへフォールバック')
    train_parser.add_argument('--cuopt-method', choices=['barrier', 'pdlp', 'concurrent', 'dual simplex'],
                             default='pdlp', help='cuOpt LP法。GEMでは独立行PDLP+crossoverを推奨')
    train_parser.add_argument('--fba-mode', choices=['separate', 'joint', 'cooperative'], default=None,
                             help='FBA構成。cooperativeは共有培地・同時交差栄養・最小共通増殖を解く')
    train_parser.add_argument('--gpu-ids', type=str, default=None,
                             help='並列環境に割り当てるGPU ID（例: 0,1,2）。DFBA_GPU_IDSでも指定可能')
    train_parser.add_argument('--gpu-slots-per-device', type=int, default=1,
                             help='1 GPUへ同時に割り当てる論理環境slot数（MIGではなくプロセス並列）')
    train_parser.add_argument('--surrogate-dir', type=str, default=None,
                             help='train_fba_surrogate.py が生成したartifactディレクトリ')
    train_parser.add_argument('--surrogate-device', choices=['cpu', 'cuda', 'auto'], default='cuda',
                             help='FBAサロゲート推論デバイス')
    train_parser.add_argument('--surrogate-batch-window-ms', type=float, default=2.0,
                             help='複数環境のGPU要求を束ねる最大待ち時間(ms)')
    train_parser.add_argument('--surrogate-max-batch-size', type=int, default=64,
                             help='GPU FBAサロゲートの最大マイクロバッチ')
    train_parser.add_argument('--surrogate-audit-interval', type=int, default=128,
                             help='各workerで厳密HiGHS監査を行うFBA呼出し間隔（0で無効）')
    train_parser.add_argument('--surrogate-objective-rtol', type=float, default=0.02,
                             help='厳密監査で許容する目的値相対誤差')
    train_parser.add_argument('--surrogate-ood-threshold', type=float, default=8.0,
                             help='学習分布からの標準化距離上限')
    
    # 評価モード
    eval_parser = subparsers.add_parser('evaluate', help='訓練済みエージェントを評価')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='訓練済みモデルのパス')
    eval_parser.add_argument('--sbml-dir', type=str, help='SBMLファイルのディレクトリ')
    eval_parser.add_argument('--max-steps', type=int, default=200,
                            help='1エピソードの最大ステップ数')
    eval_parser.add_argument('--target-degradation', type=float, default=0.9,
                            help='目標ゴム分解率')
    eval_parser.add_argument('--amino-acid-cost', type=float, default=0.1,
                            help='アミノ酸コスト係数')
    eval_parser.add_argument('--eval-episodes', type=int, default=10,
                            help='評価エピソード数')
    eval_parser.add_argument('--solver-backend', choices=['glpk', 'highs', 'cuopt', 'auto', 'surrogate'], default='highs',
                            help='FBAソルバー。cuOptはNVIDIA GPU用、autoは未導入時にGLPKへフォールバック')
    eval_parser.add_argument('--cuopt-method', choices=['barrier', 'pdlp', 'concurrent', 'dual simplex'],
                            default='pdlp', help='cuOpt LP法')
    eval_parser.add_argument('--fba-mode', choices=['separate', 'joint', 'cooperative'], default=None,
                            help='FBA構成。cooperativeは共有培地・同時交差栄養・最小共通増殖を解く')
    eval_parser.add_argument('--surrogate-dir', type=str, default=None)
    eval_parser.add_argument('--surrogate-device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    eval_parser.add_argument('--surrogate-audit-interval', type=int, default=1,
                            help='評価時は既定で毎回HiGHS監査')
    eval_parser.add_argument('--surrogate-objective-rtol', type=float, default=0.02)
    eval_parser.add_argument('--surrogate-ood-threshold', type=float, default=8.0)

    for command_parser in (train_parser, eval_parser):
        command_parser.add_argument('--consortium-profile', choices=['legacy3', 'pf-helper3', 'or16-ns21'], default='legacy3')
        command_parser.add_argument('--physiology-config', help='JSON with explicit, uncalibrated maintenance/death assumptions')
        command_parser.add_argument('--dynamics', choices=['legacy', 'audited', 'physiology'], default='legacy',
                                    help='audited requires a newly trained policy and exact CPU separate FBA')
        command_parser.add_argument('--control-dt', type=float, default=.2, help='Controller interval (hours)')
        command_parser.add_argument('--internal-dt', type=float, default=.025, help='Maximum audited integration interval (hours)')
        command_parser.add_argument('--initial-nh4', type=float, default=None, help='Initial NH4 (mmol/L)')
        command_parser.add_argument('--initial-rubber', type=float, default=100., help='Initial rubber (g/L)')
        command_parser.add_argument('--ph-target', type=float, default=None)
        command_parser.add_argument('--observation-schema', choices=['legacy_v1', 'metabolic_v2'], default=None)
        command_parser.add_argument('--max-specific-feed-rate', type=float, default=.1, help='audited_rates_v2 individual pump maximum (mmol/L/h)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'evaluate':
        evaluate_agent(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    # Ctrl+Cで子プロセスを確実に終了させるシグナルハンドラ
    def _cleanup_handler(signum, frame):
        print('\n🛑 学習を安全に停止中...')
        # 全子プロセスにSIGTERMを送信
        import multiprocessing
        for p in multiprocessing.active_children():
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)
    main()
