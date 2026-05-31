import subprocess
import os
import sys
import yaml

# Assume the project root is the parent of the scripts folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTING_PATH = os.path.join(PROJECT_ROOT, "setting.yaml")

def load_config():
    """Loads the project settings from yaml."""
    if not os.path.exists(SETTING_PATH):
        print(f"[ERROR] setting.yaml not found at {SETTING_PATH}")
        sys.exit(1)
    with open(SETTING_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_script(script_name):
    """Runs a python script from the src directory."""
    script_path = os.path.join(PROJECT_ROOT, "src", script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
        
    cmd = [sys.executable, script_path]
    print(f"\n[EXEC] {' '.join(cmd)}")
    
    # Scripts now read ../setting.yaml, so they MUST be run with CWD = src/
    # or we ensure they can find it. Since they use "../setting.yaml",
    # they expect to be run from inside the 'src' directory.
    result = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "src"))
    
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed with return code {result.returncode}")
        return False
    return True

CONFIG = load_config()

def main():
    
    # Available steps mapping to script filenames
    STEP_MAP = {
        "extract": "extract_data.py",
        "convert": "convert_pt.py",
        "freeze": "freezing_dataset.py",
        "manifest": "pt_file_list.py",
        "train": "train_model.py"
    }

    WAY = CONFIG["way"]
    if WAY == "model":
        selected_steps = ["train"]
    elif WAY == "dataset":
        selected_steps = ["extract", "convert", "manifest", "freeze"]
    else:
        selected_steps = ["extract", "convert", "manifest", "freeze", "train"]

    print("="*60)
    print("  PIPELINE START (Config-driven)")
    print("="*60)

    for step in selected_steps:
        script_file = STEP_MAP[step]
        print(f"\n>>> Step: {step.upper()}")
        if not run_script(script_file):
            print(f"\n[CRITICAL] Pipeline aborted due to error in {step}.")
            sys.exit(1)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()