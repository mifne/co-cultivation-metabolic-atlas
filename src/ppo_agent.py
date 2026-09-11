"""
PPO Agent for Consortium Control
Proximal Policy Optimization エージェント
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any, Callable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import warnings


class ConsortiumPPOAgent:
    """コンソーシアム制御用PPOエージェント"""
    
    def __init__(
        self,
        env,
        learning_rate: float | Callable[[float], float] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        verbose: int = 1,
        n_envs: int = 1,  # 並列環境数
        tensorboard_log: Optional[str] = None,
        subproc_start_method: Optional[str] = None,
        target_kl: Optional[float] = None,
    ):
        """
        Args:
            env: Gymnasium環境
            learning_rate: 学習率
            n_steps: 各環境でのステップ数
            batch_size: ミニバッチサイズ
            n_epochs: 各更新でのエポック数
            gamma: 割引率
            gae_lambda: GAEのλ
            clip_range: PPOクリッピング範囲
            ent_coef: エントロピー係数
            vf_coef: 価値関数係数
            max_grad_norm: 勾配クリッピング
            device: 'auto', 'cpu', 'cuda'
            verbose: ログレベル
            tensorboard_log: TensorBoardログの保存先
            target_kl: 方策更新のKL停止基準。Noneは従来どおり無効。
        """
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be finite in [0, 1]')
        if target_kl is not None and (not np.isfinite(target_kl) or target_kl <= 0):
            raise ValueError('target_kl must be positive and finite, or None')
        # --- 環境をベクトル化（並列化対応） ---
        if isinstance(env, list):
            # 関数のリストが渡された場合（SubprocVecEnv）
            self.env = SubprocVecEnv(env, start_method=subproc_start_method)
            self.n_envs = len(env)
        else:
            # 単一の環境（関数またはインスタンス）が渡された場合
            env_fn = env if callable(env) else lambda: env
            self.env = DummyVecEnv([env_fn])
            self.n_envs = 1
        
        # --- 数学的安定化: VecNormalize の導入 ---
        # Reward statistics must use the same return horizon as the PPO value
        # target. VecNormalize otherwise silently keeps its default gamma=.99.
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True,
                                clip_obs=10., gamma=gamma)
        self.policy_contract = self._read_policy_contract()
        
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
        # SB3 includes this attribute in every ZIP, including CheckpointCallback
        # saves; array shape alone cannot distinguish different organisms or
        # different physical meanings of otherwise identical control vectors.
        self.model.cultivation_policy_contract = self.policy_contract
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'rubber_degradation': [],
            'diversity_scores': []
        }

    def _read_policy_contract(self):
        try:
            contracts = self.env.env_method('get_policy_contract')
        except AttributeError:
            return None  # General Gym test/example environments have no contract.
        encoded = [json.dumps(c, sort_keys=True, allow_nan=False) for c in contracts]
        if len(set(encoded)) != 1:
            raise ValueError('PPO environments have different cultivation policy contracts')
        # Freeze by value: live metadata dictionaries must not mutate the
        # checkpoint's meaning through a shared Python reference.
        return json.loads(encoded[0])

    def _assert_policy_contract_unchanged(self):
        if self._read_policy_contract() != self.policy_contract:
            raise ValueError('Cultivation configuration changed after policy construction; '
                             'create a compatible policy/environment before proceeding')
    
    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        save_freq: int = 10000,
        save_path: str = 'outputs/checkpoints',
        eval_freq: int = 5000,
        eval_env = None,
        n_eval_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        エージェントを訓練
        
        Args:
            total_timesteps: 総訓練ステップ数
            callback: カスタムコールバック
            log_interval: ログ出力間隔
            save_freq: チェックポイント保存頻度
            save_path: チェックポイント保存先
            eval_freq: 評価頻度
            eval_env: 評価用環境
            n_eval_episodes: 評価エピソード数
        
        Returns:
            訓練履歴
        """
        print(f"🎓 訓練開始: {total_timesteps:,}ステップ")
        self._assert_policy_contract_unchanged()
        
        # コールバックのリスト
        callbacks = []
        
        # カスタムコールバック
        training_callback = TrainingCallback(self.training_history)
        callbacks.append(training_callback)
        
        # チェックポイント保存
        checkpoint_path = Path(save_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // self.n_envs,
            save_path=str(checkpoint_path),
            name_prefix='ppo_consortium',
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
        
        # 評価コールバック（オプション）
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(checkpoint_path / 'best_model'),
                log_path=str(checkpoint_path / 'eval_logs'),
                eval_freq=eval_freq // self.n_envs,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # ユーザー指定のコールバックを追加
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)
        
        # 訓練実行
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            reset_num_timesteps=False
        )
        
        # 訓練履歴を可視化
        self.plot_training_history(save_path=checkpoint_path / 'training_curves.png')
        print("✅ 訓練完了")
        
        return self.training_history
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        行動を予測
        
        Args:
            observation: Saved VecNormalizeで正規化済みの観測。生観測はpredict_rawを使用。
            deterministic: 決定的な行動を選択するか
        
        Returns:
            (action, state)
        """
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return action, state

    def predict_raw(self, observation: np.ndarray, deterministic: bool = True):
        """Infer from an unnormalized Gym observation without updating statistics."""
        normalized = self.env.normalize_obs(np.asarray(observation))
        return self.predict(normalized, deterministic=deterministic)
    
    def save(self, path: str):
        """
        モデルと正規化統計を保存
        
        Args:
            path: 保存先パス
        """
        save_path = Path(path)
        self._assert_policy_contract_unchanged()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        if self.policy_contract is not None:
            save_path.with_suffix('.contract.json').write_text(
                json.dumps(self.policy_contract, indent=2, sort_keys=True, allow_nan=False),
                encoding='utf-8')
        
        # VecNormalizeの統計を保存 (拡張子 .pkl)
        stats_path = str(save_path.with_suffix('.pkl'))
        self.env.save(stats_path)
        
        print(f"💾 モデル保存: {save_path}")
        print(f"📊 正規化統計保存: {stats_path}")
    
    def load(self, path: str):
        """
        モデルと正規化統計を読み込み
        
        Args:
            path: モデルファイルパス
        """
        # 現在のTensorBoardログパスを保持
        self._assert_policy_contract_unchanged()
        current_tb_log = self.model.tensorboard_log
        current_device = self.model.device
        
        # VecNormalizeの統計を読み込み
        path_obj = Path(path)
        stats_path = path_obj.with_suffix('.pkl')
        
        # CheckpointCallbackの命名規則に対応: prefix + "_vecnormalize_" + steps + "_steps.pkl"
        # ここでは単純に .zip を除去して _vecnormalize_ を含むファイルを検索する
        checkpoint_stats_path = path_obj.parent / path_obj.name.replace("ppo_consortium", "ppo_consortium_vecnormalize").replace(".zip", ".pkl")

        if stats_path.exists():
            load_path = stats_path
        elif checkpoint_stats_path.exists():
            load_path = checkpoint_stats_path
        else:
            raise FileNotFoundError(
                f'Saved VecNormalize statistics are required for {path}; '
                f'expected {stats_path} or {checkpoint_stats_path}')

        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(path, device='cpu')
        saved_contract = data.get('cultivation_policy_contract')
        if saved_contract is None and self.policy_contract is not None:
            if self.policy_contract['control']['schema'] != 'legacy_v1':
                raise ValueError('Checkpoint has no cultivation contract; cannot use it with audited controls')
            warnings.warn('Legacy checkpoint has no cultivation contract; organism and control semantics '
                          'cannot be verified. Use only its original legacy environment.', RuntimeWarning)
        elif saved_contract != self.policy_contract:
            raise ValueError('Checkpoint cultivation contract differs from this environment')

        # Load only after checking both normalization and physical semantics.
        self.model = PPO.load(path, env=self.env, tensorboard_log=current_tb_log,
                              device=current_device)

        if load_path:
            # 現在の venv を使って VecNormalize をロード
            self.env = VecNormalize.load(str(load_path), self.env.venv)
            # 再構築した env をモデルに再接続
            self.model.set_env(self.env)
            print(f"📂 正規化統計読み込み: {load_path}")
        
        print(f"📂 モデル読み込み: {path}")
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        訓練履歴を可視化
        
        Args:
            save_path: 保存先パス（Noneの場合は表示のみ）
        """
        if not self.training_history['episode_rewards']:
            print("⚠️  訓練履歴が空です")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Training History', fontsize=16)
        
        # 1. エピソード報酬
        ax = axes[0, 0]
        rewards = self.training_history['episode_rewards']
        ax.plot(rewards, alpha=0.6, label='Episode Reward')
        # 移動平均
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. エピソード長
        ax = axes[0, 1]
        lengths = self.training_history['episode_lengths']
        ax.plot(lengths, alpha=0.6, label='Episode Length')
        if len(lengths) > 10:
            window = min(50, len(lengths) // 10)
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(lengths)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. ゴム分解率（利用可能な場合）
        ax = axes[1, 0]
        if self.training_history['rubber_degradation']:
            degradation = self.training_history['rubber_degradation']
            ax.plot(degradation, alpha=0.6, label='Rubber Remaining')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rubber Concentration (g/L)')
            ax.set_title('Rubber Degradation Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No rubber degradation data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 4. 報酬の分布
        ax = axes[1, 1]
        if len(rewards) > 0:
            ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax.axvline(np.median(rewards), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 訓練曲線: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        エージェントを評価
        
        Args:
            n_episodes: 評価エピソード数
            deterministic: 決定的な行動を選択するか
        
        Returns:
            評価結果の統計
        """
        print(f"📊 評価: {n_episodes}エピソード")
        self._assert_policy_contract_unchanged()
        if type(n_episodes) is not int or n_episodes < 1:
            raise ValueError('n_episodes must be a positive integer')
        
        episode_rewards = []
        episode_lengths = []
        rubber_degradations = []
        
        old_training, old_norm_reward = self.env.training, self.env.norm_reward
        self.env.training, self.env.norm_reward = False, False
        try:
            for episode in range(n_episodes):
                obs = self.env.reset()
                episode_reward = 0.
                episode_length = 0
                try:
                    initial_rubber = float(self.env.get_attr('initial_rubber', indices=[0])[0])
                except AttributeError:
                    initial_rubber = None
                while True:
                    action, _ = self.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = self.env.step(action)
                    # Aggregate one declared lane; never read state after the
                    # VecEnv terminal auto-reset for an endpoint measurement.
                    episode_reward += float(reward[0])
                    episode_length += 1
                    if initial_rubber is None:
                        if 'initial_rubber_g_l' in info[0]:
                            initial_rubber = float(info[0]['initial_rubber_g_l'])
                        elif 'total_rubber_degraded' in info[0] and 'rubber_remaining' in info[0]:
                            initial_rubber = float(info[0]['rubber_remaining'] + info[0]['total_rubber_degraded'])
                    if done[0]:
                        final_rubber = info[0].get('rubber_remaining')
                        degradation = ((initial_rubber - float(final_rubber)) / initial_rubber
                                       if initial_rubber is not None and initial_rubber > 0 and final_rubber is not None
                                       else float('nan'))
                        rubber_degradations.append(degradation)
                        break
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
        finally:
            self.env.training, self.env.norm_reward = old_training, old_norm_reward
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_degradation': np.mean(rubber_degradations),
            'std_degradation': np.std(rubber_degradations)
        }
        
        print(f"📈 平均報酬: {results['mean_reward']:.2f}, "
              f"エピソード長: {results['mean_length']:.1f}, "
              f"ゴム分解率: {results['mean_degradation']:.2%}")
        
        return results


class TrainingCallback(BaseCallback):
    """訓練中のカスタムコールバック"""
    
    def __init__(self, history: Dict[str, list], verbose: int = 0):
        super().__init__(verbose)
        self.history = history
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """各ステップで呼ばれる"""
        # 全ての環境の状態をチェック
        for i, done in enumerate(self.locals.get('dones')):
            if done:
                info = self.locals.get('infos')[i]
                
                # 報酬とエピソード長を記録
                if 'episode' in info:
                    self.history['episode_rewards'].append(info['episode']['r'])
                    self.history['episode_lengths'].append(info['episode']['l'])
                
                # ゴム分解率を記録
                if 'rubber_remaining' in info:
                    self.history['rubber_degradation'].append(info['rubber_remaining'])
        
        return True
    
    def _on_training_end(self) -> None:
        """訓練終了時の処理"""
        pass
