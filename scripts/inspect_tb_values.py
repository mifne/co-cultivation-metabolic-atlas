import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

log_dir = "outputs/final_v12_vision_fix/tensorboard/PPO_0"
try:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"--- Scalar Tags Found ---")
    for t in tags:
        print(f"Tag: {t}")
    
    if 'Science/Rubber_Remaining' in tags:
        events = ea.Scalars('Science/Rubber_Remaining')
        if events:
            print(f"\n--- Science/Rubber_Remaining Values ---")
            # Print first 5 and last 5
            for e in events[:5]:
                print(f"Step {e.step}: Value {e.value}")
            if len(events) > 10:
                print("...")
                for e in events[-5:]:
                    print(f"Step {e.step}: Value {e.value}")
        else:
            print("\nScience/Rubber_Remaining has no events.")
    else:
        print("\nScience/Rubber_Remaining NOT FOUND in tags.")

except Exception as e:
    print(f"Error: {e}")
