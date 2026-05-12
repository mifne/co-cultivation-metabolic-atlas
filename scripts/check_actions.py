import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_dir = "outputs/final_kinetic_paradigm_run/tensorboard/PPO_0"
try:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Tags found: {tags}")
    
    # In my environment, I should have Reward/r_cost or something?
    # Actually, main.py might not be logging actions.
    # Let's check Reward tags
    reward_tags = [t for t in tags if t.startswith('Reward/')]
    for rt in reward_tags:
        events = ea.Scalars(rt)
        if events:
            print(f"{rt}: {events[-1].value}")

except Exception as e:
    print(f"Error: {e}")

