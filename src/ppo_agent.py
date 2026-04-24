"""
PPO Agent for Consortium Control
Proximal Policy Optimization エージェント
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any, Callable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ConsortiumPPOAgent:
    """コンソーシアム制御用PPOエージェント"""
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
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
        n_envs: int = 1  # 並列環境数
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
        """
        # 環境をベクトル化（並列化対応）
        if n_envs > 1:
            print(f"  🚀 並列環境を使用: {n_envs}プロセス")
            # 並列環境の作成（各プロセスで独立したシミュレーター）
            # 注: envは関数として渡す必要がある
            if callable(env):
                self.env = SubprocVecEnv([env for _ in range(n_envs)])
            else:
                # envがインスタンスの場合はDummyVecEnvを使用
                print(f"  ⚠️  envがインスタンスのため、DummyVecEnvを使用")
                self.env = DummyVecEnv([lambda: env])
                n_envs = 1
        else:
            self.env = DummyVecEnv([lambda: env])
        
        self.n_envs = n_envs
        
        # PPOモデルの初期化（CPU強制）
        print(f"  💻 計算デバイス: {device} (dFBAシミュレーションがボトルネックのため)")
        print(f"  📊 並列環境数: {n_envs}")
        
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
            verbose=verbose,
            device='cpu',  # 強制的にCPU（deviceパラメータを無視）
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])  # SB3 v1.8.0以降の形式
            )
        )
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'rubber_degradation': [],
            'diversity_scores': []
        }
    
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
        print(f"🎓 PPOエージェントの訓練を開始 (総ステップ数: {total_timesteps:,})")
        
        # コールバックのリスト
        callbacks = []
        
        # カスタムコールバック
        training_callback = TrainingCallback(self.training_history)
        callbacks.append(training_callback)
        
        # チェックポイント保存
        checkpoint_path = Path(save_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // self.n_envs,  # 並列環境数で調整
            save_path=str(checkpoint_path),
            name_prefix='ppo_consortium',
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        callbacks.append(checkpoint_callback)
        print(f"  💾 チェックポイント保存: {save_freq:,}ステップごと → {checkpoint_path}")
        
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
            print(f"  📊 評価: {eval_freq:,}ステップごと ({n_eval_episodes}エピソード)")
        
        # ユーザー指定のコールバックを追加
        if callback is not None:
            callbacks.append(callback)
        
        # 訓練実行
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )
        
        print("✅ 訓練完了")
        
        # 訓練履歴を可視化
        self.plot_training_history(save_path=checkpoint_path / 'training_curves.png')
        
        return self.training_history
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        行動を予測
        
        Args:
            observation: 観測
            deterministic: 決定的な行動を選択するか
        
        Returns:
            (action, state)
        """
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return action, state
    
    def save(self, path: str):
        """
        モデルを保存
        
        Args:
            path: 保存先パス
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        print(f"💾 モデル保存: {save_path}")
    
    def load(self, path: str):
        """
        モデルを読み込み
        
        Args:
            path: モデルファイルパス
        """
        self.model = PPO.load(path, env=self.env)
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
            print(f"  📊 訓練曲線を保存: {save_path}")
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
        print(f"📊 エージェント評価開始 ({n_episodes}エピソード)")
        
        episode_rewards = []
        episode_lengths = []
        rubber_degradations = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            initial_rubber = None
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if initial_rubber is None:
                    initial_rubber = info[0].get('rubber_remaining', 0)
                
                if done[0]:
                    final_rubber = info[0].get('rubber_remaining', 0)
                    degradation = (initial_rubber - final_rubber) / initial_rubber if initial_rubber > 0 else 0
                    rubber_degradations.append(degradation)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  エピソード {episode + 1}: 報酬={episode_reward:.2f}, "
                  f"長さ={episode_length}, ゴム分解率={degradation:.2%}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_degradation': np.mean(rubber_degradations),
            'std_degradation': np.std(rubber_degradations)
        }
        
        print(f"\n📈 評価結果:")
        print(f"  平均報酬: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  平均エピソード長: {results['mean_length']:.1f}")
        print(f"  平均ゴム分解率: {results['mean_degradation']:.2%} ± {results['std_degradation']:.2%}")
        
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
        # エピソード終了時の処理
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            
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
        print(f"\n📊 訓練統計:")
        if self.history['episode_rewards']:
            print(f"  総エピソード数: {len(self.history['episode_rewards'])}")
            print(f"  平均報酬: {np.mean(self.history['episode_rewards']):.2f}")
            print(f"  最大報酬: {np.max(self.history['episode_rewards']):.2f}")
