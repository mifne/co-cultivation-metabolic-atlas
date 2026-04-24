"""
PPO Agent for Consortium Control
Proximal Policy Optimization エージェント
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path


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
        device: str = 'cpu',  # 強制的にCPU使用（dFBAがボトルネックのためGPU効果なし）
        verbose: int = 1
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
        # 環境をベクトル化
        self.env = DummyVecEnv([lambda: env])
        
        # PPOモデルの初期化（CPU強制）
        print(f"  💻 計算デバイス: {device} (dFBAシミュレーションがボトルネックのため)")
        
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
        log_interval: int = 10
    ) -> Dict[str, Any]:
        """
        エージェントを訓練
        
        Args:
            total_timesteps: 総訓練ステップ数
            callback: カスタムコールバック
            log_interval: ログ出力間隔
        
        Returns:
            訓練履歴
        """
        print(f"🎓 PPOエージェントの訓練を開始 (総ステップ数: {total_timesteps})")
        
        # カスタムコールバックの設定
        if callback is None:
            callback = TrainingCallback(self.training_history)
        
        # 訓練実行
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
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
