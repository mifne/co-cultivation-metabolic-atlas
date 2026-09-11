import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_pha_rewards(log_dir):
    try:
        event_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(log_dir) for f in filenames if 'events.out.tfevents' in f]
        if not event_files: return "No event files found."
        event_files.sort(key=os.path.getmtime, reverse=True)
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        res = []
        if 'Reward/r_pha' in ea.scalars.Keys():
            events = ea.scalars.Items('Reward/r_pha')
            # get the latest 5 values
            for e in events[-5:]:
                res.append(f"Step {e.step}: r_pha = {e.value:.4f}")
            return "\n".join(res)
        return "Key 'Reward/r_pha' not found."
    except Exception as e:
        return str(e)

print(get_pha_rewards('outputs/tensorboard/'))
