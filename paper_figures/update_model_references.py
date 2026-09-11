import glob
import os

def main():
    print("Locating python scripts in paper_figures/...")
    scripts = glob.glob("paper_figures/*.py") + [
        "scripts/extract_final_fig11.py",
        "scripts/run_information_deprivation.py"
    ]
    
    old_zip1 = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    old_pkl1 = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    old_zip2 = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    old_pkl2 = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    
    new_zip = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    new_pkl = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    
    replacements = {
        old_zip1: new_zip,
        old_pkl1: new_pkl,
        old_zip2: new_zip,
        old_pkl2: new_pkl
    }
    
    updated_count = 0
    for script in scripts:
        if not os.path.exists(script):
            continue
        with open(script, "r", encoding="utf-8") as f:
            content = f.read()
            
        modified = False
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
                modified = True
                
        if modified:
            with open(script, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Updated: {script}")
            updated_count += 1
            
    print(f"Successfully updated model references in {updated_count} files.")

if __name__ == "__main__":
    main()
