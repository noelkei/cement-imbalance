import os
import yaml
import itertools
import shutil
from pathlib import Path
from copy import deepcopy
from training.utils import ROOT_PATH
import yaml
from itertools import product
import time
import traceback

from training.train_flow_pre import train_flowpre_pipeline  # assumes your pipeline is available here

CONFIG_BASE_PATH = ROOT_PATH / "config" / "flow_pre.yaml"
CONFIG_TMP_DIR = ROOT_PATH / "config" / "tmp_grid"
CONFIG_TMP_DIR.mkdir(parents=True, exist_ok=True)

BASE_NAME = "VM"

STATUS_FILE = ROOT_PATH / "config" / "grid_status.yaml"
MAX_RETRIES = 3
RETRY_WAIT = 5  # seconds

GRID = {
    "training.learning_rate": [1e-3, 1e-4, 1e-5],
    "model.hidden_features": [128, 192, 256],
    "model.num_layers": [2, 3, 4],
    "model.affine_rq_ratio,model.final_rq_layers": [
        ([1, 3], 3),
        ([1, 5], 5),
        ([1, 6], 6),
        ([1, 8], 8)
    ],
    "training.use_mean_penalty,training.use_std_penalty": [
        (False, False),
        (True, True)
    ],
    "training.use_skew_penalty,training.use_kurtosis_penalty": [
        (False, False),
        (True, True)
    ]
}

def generate_param_combinations(grid):
    expanded = []
    for key, values in grid.items():
        keys = key.split(",")
        expanded.append([(keys, v if isinstance(v, tuple) else (v,)) for v in values])
    for combo in itertools.product(*expanded):
        flat = {}
        for keys, values in combo:
            for k, v in zip(keys, values):
                flat[k] = v
        yield flat

def update_nested_dict(d, key_path, value):
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def create_and_save_config(base_path, param_updates, version_id):
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)
    for key, val in param_updates.items():
        update_nested_dict(config, key, val)
    config_path = CONFIG_TMP_DIR / f"flow_pre_{version_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path

def param_suffix(params):
    def sanitize(val):
        if isinstance(val, list):
            return "x".join(map(str, val))
        return str(val).replace(".", "")
    return "_".join(f"{k.split('.')[-1]}{sanitize(v)}" for k, v in sorted(params.items()))

def print_all_configs():
    combinations = list(generate_param_combinations(GRID))
    print(f"🧾 Generating and printing {len(combinations)} configs...")

    for i, param_set in enumerate(combinations, 1):
        suffix = param_suffix(param_set)
        version_name = f"{BASE_NAME}_{i:03d}_{suffix}"
        config_path = create_and_save_config(CONFIG_BASE_PATH, param_set, version_id=version_name)

        print(f"\n🛠️  Config for: {version_name}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        for line in config_str.splitlines():
            print(f"   {line}")

def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_status(status):
    tmp_path = STATUS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(status, f)
    tmp_path.replace(STATUS_FILE)

def run_training_with_retry(version_name, config_path, retries=MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            train_flowpre_pipeline(
                condition_col="type",
                config_filename=config_path,
                base_name=version_name,
                device="cpu",
                verbose=False
            )
            return True
        except Exception as e:
            print(f"❌ Attempt {attempt} failed for {version_name}")
            traceback.print_exc()
            if attempt < retries:
                print(f"🔁 Retrying in {RETRY_WAIT} sec...")
                time.sleep(RETRY_WAIT)
    return False

def main():
    combinations = list(generate_param_combinations(GRID))
    print(f"🔍 Total runs: {len(combinations)}")

    status = load_status()

    # Initial pass
    for i, param_set in enumerate(combinations, 1):
        suffix = param_suffix(param_set)
        version_name = f"{BASE_NAME}_{i:03d}_{suffix}"

        if status.get(version_name) == "done":
            print(f"✅ Skipping already completed: {version_name}")
            continue

        print(f"\n🚀 [{i}/{len(combinations)}] Training {version_name}")
        config_path = create_and_save_config(CONFIG_BASE_PATH, param_set, version_id=version_name)

        success = run_training_with_retry(version_name, config_path)
        status[version_name] = "done" if success else "failed"
        save_status(status)

    # Retry failed ones at the end
    failed_versions = [v for v, s in status.items() if s == "failed"]
    if failed_versions:
        print(f"\n🔁 Retrying failed runs: {len(failed_versions)}")

        for version_name in failed_versions:
            print(f"\n🔥 Retrying {version_name}")
            param_idx = int(version_name.split("_")[2]) - 1
            param_set = combinations[param_idx]
            config_path = create_and_save_config(CONFIG_BASE_PATH, param_set, version_id=version_name)

            success = run_training_with_retry(version_name, config_path)
            status[version_name] = "done" if success else "failed"
            save_status(status)

    print("\n✅ All done.")

if __name__ == "__main__":
    main()
