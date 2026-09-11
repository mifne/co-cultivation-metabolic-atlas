import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ConsortiumCallback(BaseCallback):
    """
    科学制約を監視するためのカスタムコールバック (v3.0)
    """
    def __init__(self, verbose=0):
        super(ConsortiumCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:

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
            if "biomass_pf" in info:
                self.logger.record("Science/Biomass_Pf", info["biomass_pf"])
            for species_name, biomass in info.get("biomass_g_l", {}).items():
                self.logger.record(f"Science/Biomass/{species_name}", biomass)
            if "pha_g_l" in info:
                self.logger.record("Science/PHA_g_l", info["pha_g_l"])
            if "r_deg" in info:
                self.logger.record("Reward/r_deg", info["r_deg"])
            if "r_surv" in info:
                self.logger.record("Reward/r_surv", info["r_surv"])
            if "r_pha" in info:
                self.logger.record("Reward/r_pha", info["r_pha"])
            if "r_cost" in info:
                self.logger.record("Reward/r_cost", info["r_cost"])
                
        return True
