import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ConsortiumCallback(BaseCallback):
    """
    科学制約とカリキュラム学習を監視・制御するためのカスタムコールバック
    """
    def __init__(self, verbose=0):
        super(ConsortiumCallback, self).__init__(verbose)
        self.total_timesteps = 0

    def _on_training_start(self) -> None:
        """訓練開始時に総ステップ数を取得し、環境に設定"""
        self.total_timesteps = self.locals.get('total_timesteps', 100000)
        if hasattr(self.training_env, 'env_method'):
            try:
                self.training_env.env_method('set_total_timesteps', self.total_timesteps)
            except Exception:
                pass

    def _on_step(self) -> bool:
        # 環境の進捗を更新
        if hasattr(self.training_env, 'env_method'):
            try:
                self.training_env.env_method('set_current_total_steps', self.num_timesteps)
            except Exception:
                pass

        # カリキュラム段階の可視化
        progress = self.num_timesteps / max(1, self.total_timesteps)
        if progress < 0.1: stage = 1
        elif progress < 0.3: stage = 2
        elif progress < 0.5: stage = 3
        elif progress < 0.8: stage = 4
        else: stage = 5
        self.logger.record("Science/Curriculum_Stage", stage)

        # 各並列環境の情報を収集
        infos = self.locals.get("infos")
        if infos:
            info = infos[0]
            if "rubber_remaining" in info:
                self.logger.record("Science/Rubber_Remaining", info["rubber_remaining"])
            if "total_rubber_degraded" in info:
                self.logger.record("Science/Total_Rubber_Degraded", info["total_rubber_degraded"])
            if "survival_hours" in info:
                self.logger.record("Science/Survival_Hours", info["survival_hours"])
            if "ph" in info:
                self.logger.record("Science/pH", info["ph"])
            if "do" in info:
                self.logger.record("Science/DO", info["do"])
            if "total_pha" in info:
                self.logger.record("Science/Total_PHA", info["total_pha"])
            if "biosurfactant" in info:
                self.logger.record("Science/Biosurfactant", info["biosurfactant"])
            if "biomass_or16" in info:
                self.logger.record("Science/Biomass_OR16", info["biomass_or16"])
            if "biomass_ns21" in info:
                self.logger.record("Science/Biomass_NS21", info["biomass_ns21"])
            if "biomass_lp" in info:
                self.logger.record("Science/Biomass_LP", info["biomass_lp"])
            if "r_deg" in info:
                self.logger.record("Reward/r_deg", info["r_deg"])
            if "r_surv" in info:
                self.logger.record("Reward/r_surv", info["r_surv"])
            if "r_pha" in info:
                self.logger.record("Reward/r_pha", info["r_pha"])
            if "r_cost" in info:
                self.logger.record("Reward/r_cost", info["r_cost"])
                
        return True
