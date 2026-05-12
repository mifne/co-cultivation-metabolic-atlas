import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_dir = "outputs/final_v15_do_monitoring/tensorboard/PPO_0"
try:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"--- Final Training Results (2,000,000 steps) ---")
    
    important_tags = [
        'Science/Rubber_Remaining',
        'Science/Total_Rubber_Degraded',
        'Science/Survival_Hours',
        'Science/pH',
        'Science/DO',
        'Science/Total_PHA',
        'Science/Biomass_OR16',
        'Science/Biomass_NS21',
        'Science/Biomass_LP'
    ]
    
    for tag in important_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                last_val = events[-1].value
                # Get max/min where appropriate
                all_vals = [e.value for e in events]
                max_val = max(all_vals)
                min_val = min(all_vals)
                
                if 'Rubber_Remaining' in tag:
                    print(f"{tag}: Final={last_val:.2f}, Min={min_val:.2f}")
                elif 'Survival_Hours' in tag:
                    print(f"{tag}: Final={last_val:.2f}, Max={max_val:.2f}")
                else:
                    print(f"{tag}: Final={last_val:.4f}")
        else:
            print(f"{tag}: NOT FOUND")

except Exception as e:
    print(f"Error: {e}")
