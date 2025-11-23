# optuna_mlp.py
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import yaml
import optuna
from optuna.trial import Trial

from training.train_mlp import train_mlp_pipeline
from training.utils import ROOT_PATH
from training.utils import load_scaled_sets, list_scaled_sets


# ──────────────────────────────────────────────────────────────────────────────
# YAML / dict utils
# ──────────────────────────────────────────────────────────────────────────────
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"YAML at {path} is not a dict.")
    return d


def _write_yaml(d: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = d
    for k in dotted_key.split(".")[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[dotted_key.split(".")[-1]] = value


def _get_by_path(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for tok in dotted.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return None
    return cur


def _trial_tag(trial: Trial) -> str:
    return f"t{trial.number:04d}"


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ──────────────────────────────────────────────────────────────────────────────
# Device resolution (CUDA / MPS / CPU) + note for MPS float32 policy
# ──────────────────────────────────────────────────────────────────────────────
def resolve_device(requested: str = "cuda") -> str:
    req = (requested or "cuda").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[optuna] ⚠️ CUDA requested but not available → falling back to MPS/CPU.")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        print("[optuna] ⚠️ MPS requested but not available → falling back to CUDA/CPU.")
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Param space (adjust to taste)
# ──────────────────────────────────────────────────────────────────────────────
def suggest_mlp_params(trial: Trial) -> Dict[str, Any]:
   model = {
       "embedding_dim": trial.suggest_int("embedding_dim", 6, 96, step=6),
       "hidden_dim":    trial.suggest_int("hidden_dim", 64, 512, step=32),
       "num_layers":    trial.suggest_int("num_layers", 1, 6),
       "activation":    trial.suggest_categorical("activation", ["relu", "gelu", "silu", "elu"]),
       "dropout":       trial.suggest_float("dropout", 0.0, 0.5),
       "batchnorm":     trial.suggest_categorical("batchnorm", [False, True]),
       "use_weight_norm": trial.suggest_categorical("use_weight_norm", [True, False]),
       "residual":        trial.suggest_categorical("residual", [True, False]),
       "final_activation": None,
       "task": "regression",
       "context_mode": trial.suggest_categorical("context_mode", ["embed", "onehot"]),
   }


   training = {
       "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
       "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
       "num_epochs":    trial.suggest_int("num_epochs", 60, 500, step=20),


       "optimizer":     trial.suggest_categorical("optimizer", ["adamw", "adam"]),
       "weight_decay":  trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
       "betas":         [0.9, 0.999],
       "eps":           1e-8,


       "lr_scheduler": "plateau",
       "lr_decay_factor": trial.suggest_float("lr_decay_factor", 0.25, 0.75),
       "lr_decay_patience": trial.suggest_int("lr_decay_patience", 15, 30),
       "lr_cooldown": 0,
       "min_lr": 0.0,
       "plateau_threshold": 1e-4,
       "plateau_threshold_mode": "rel",


       "early_stopping_patience": trial.suggest_int("early_stopping_patience", 20, 50),
       "early_stop_min_delta": trial.suggest_float("early_stop_min_delta", 0.0, 5e-3),


       "loss_reduction":"per_class_equal",
       "regression_group_metric": trial.suggest_categorical("regression_group_metric", ["mse", "rmse", "rrmse"]),


       "save_model": False,
       "save_states": False,
       "save_results": True,
       "log_training": False,
   }
   return {"model": model, "training": training}


# ──────────────────────────────────────────────────────────────────────────────
# Config assembly per trial
# ──────────────────────────────────────────────────────────────────────────────
def build_trial_config(
    base_cfg: Dict[str, Any],
    trial_params: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    num_epochs_override: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {**base_cfg}
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("interpretability", {})

    # merge model/training
    for k, v in trial_params.get("model", {}).items():
        cfg["model"][k] = v
    for k, v in trial_params.get("training", {}).items():
        cfg["training"][k] = v

    if seed is not None:
        cfg["seed"] = int(seed)
        _deep_set(cfg, "training.seed", int(seed))

    if num_epochs_override is not None:
        _deep_set(cfg, "training.num_epochs", int(num_epochs_override))

    if extra_overrides:
        for dotted, value in extra_overrides.items():
            _deep_set(cfg, dotted, value)

    return cfg


def write_trial_config(cfg: Dict[str, Any], trial: Trial, config_dir: Path | None = None) -> Tuple[str, Path]:
    if config_dir is None:
        config_dir = Path(ROOT_PATH) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mlp_optuna_{_trial_tag(trial)}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    out_path = config_dir / f"{safe}.yaml"
    _write_yaml(cfg, out_path)
    return out_path.name, out_path


# ──────────────────────────────────────────────────────────────────────────────
# Metric pick
# ──────────────────────────────────────────────────────────────────────────────
def select_val_metric(val_metrics: Dict[str, Any], metric_path: str) -> float:
    """
    metric_path examples:
      "overall.rmse" | "overall.mse" | "overall.rrmse"
      "per_class.rrmse"         -> sum over classes of per-class["rrmse"]
      "per_class.0.rmse"        -> metric for a specific class (here class 0)
    """
    toks = (metric_path or "").split(".")
    if not toks:
        raise KeyError("Empty metric_path.")

    head = toks[0]

    if head == "overall":
        if len(toks) != 2:
            raise KeyError(f"Use 'overall.<metric>', got '{metric_path}'.")
        metric = toks[1]
        value = val_metrics.get("overall", {}).get(metric, None)

    elif head == "per_class":
        if len(toks) == 2:
            # per_class.<metric>  → sum over all classes
            metric = toks[1]
            pc = val_metrics.get("per_class", {})
            if not isinstance(pc, dict) or not pc:
                raise KeyError("val_metrics['per_class'] missing or empty.")
            total = 0.0
            for cls_id, d in pc.items():
                if metric not in d:
                    raise KeyError(f"Metric '{metric}' missing for class '{cls_id}'.")
                total += float(d[metric])
            value = total

        elif len(toks) == 3:
            # per_class.<cls>.<metric>  → single class
            cls_str, metric = toks[1], toks[2]
            try:
                cls_id = int(cls_str)
            except ValueError:
                raise KeyError(f"Class id must be int, got '{cls_str}'.")
            pc = val_metrics.get("per_class", {})
            value = pc.get(cls_id, {}).get(metric, None)
        else:
            raise KeyError(f"Use 'per_class.<metric>' or 'per_class.<cls>.<metric>', got '{metric_path}'.")

    else:
        # Fallback: old dotted lookup (kept for backward-compat)
        value = _get_by_path(val_metrics, metric_path)

    if value is None:
        raise KeyError(f"Metric path '{metric_path}' not found in val_metrics.")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"Metric at '{metric_path}' is not finite: {v}")
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Splits preview / validation printed at study START
# ──────────────────────────────────────────────────────────────────────────────
def _preview_and_validate_splits(
    *,
    X_train: Optional[pd.DataFrame],
    X_val:   Optional[pd.DataFrame],
    X_test:  Optional[pd.DataFrame],
    y_train: Optional[pd.DataFrame],
    y_val:   Optional[pd.DataFrame],
    y_test:  Optional[pd.DataFrame],
    condition_col: str,
    base_cfg: Dict[str, Any],
) -> None:
    print("\n" + "=" * 100)
    print("🔎 Optuna Study — Input data sanity check")
    tnames = base_cfg.get("model", {}).get("target_names", None)
    print(f"• expected condition_col: '{condition_col}'")
    print(f"• configured target_names (if provided): {tnames}")
    print("• Note: if splits are not provided, the training loader will build them internally.\n")

    def _check_xy(X: Optional[pd.DataFrame], y: Optional[pd.DataFrame], tag: str):
        if X is None or y is None:
            print(f"  {tag}: (not provided) → will be loaded by pipeline.")
            return
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeError(f"{tag}: X/y must be pandas DataFrames.")
        for need in ["post_cleaning_index", condition_col]:
            if need not in X.columns:
                raise ValueError(f"{tag}: X missing required column '{need}'.")
        if "post_cleaning_index" not in y.columns:
            raise ValueError(f"{tag}: y missing required column 'post_cleaning_index'.")

        # brief preview
        print(f"  {tag}:")
        print(f"    X.shape={X.shape}, y.shape={y.shape}")
        print(f"    X.cols head: {list(X.columns)[:10]}")
        print(f"    y.cols head: {list(y.columns)[:10]}")
        print("    X.head():")
        print(X.head(5).to_string(index=False))
        print("    y.head():")
        print(y.head(5).to_string(index=False))

    _check_xy(X_train, y_train, "TRAIN")
    _check_xy(X_val,   y_val,   "VAL")
    _check_xy(X_test,  y_test,  "TEST")
    print("=" * 100 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Objective
# ──────────────────────────────────────────────────────────────────────────────
def objective_mlp(
    trial: Trial,
    *,
    base_config_path: str | Path,
    base_name_prefix: str = "mlp_optuna",
    condition_col: str = "type",
    device: str = "cuda",
    metric_path: str = "overall.rmse",
    suggest_fn = suggest_mlp_params,
    seed: Optional[int] = None,
    seed_base: Optional[int] = None,   # if seed is None, use seed_base + trial.number
    num_epochs_override: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    X_train: Optional[pd.DataFrame] = None,
    X_val:   Optional[pd.DataFrame] = None,
    X_test:  Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    y_val:   Optional[pd.DataFrame] = None,
    y_test:  Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> float:
    base_cfg = _load_yaml(base_config_path)
    trial_params = suggest_fn(trial)

    # Make per-trial seed if not fixed
    trial_seed = seed if seed is not None else (None if seed_base is None else int(seed_base + trial.number))

    cfg = build_trial_config(
        base_cfg, trial_params,
        seed=trial_seed,
        num_epochs_override=num_epochs_override,
        extra_overrides=extra_overrides,
    )

    cfg_name, _ = write_trial_config(cfg, trial)
    dev = resolve_device(device)
    base_name = f"{base_name_prefix}_{_trial_tag(trial)}"

    # Train (your pipeline returns (model, val_metrics))
    model_or_tuple = train_mlp_pipeline(
        condition_col=condition_col,
        config_filename=cfg_name,
        base_name=base_name,
        device=dev,
        seed=trial_seed,
        verbose=verbose,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )

    if not (isinstance(model_or_tuple, tuple) and len(model_or_tuple) == 2):
        raise RuntimeError("train_mlp_pipeline must return (model, val_metrics).")

    _, val_metrics = model_or_tuple
    value = select_val_metric(val_metrics, metric_path)
    trial.report(value, step=0)
    return float(value)


# ──────────────────────────────────────────────────────────────────────────────
# Study runner
# ──────────────────────────────────────────────────────────────────────────────
def run_mlp_study(
    *,
    base_config_path: str | Path,
    metric_path: str = "overall.rmse",
    # Either set n_trials (fixed) OR None to use improvement-based early stop
    n_trials: Optional[int] = 50,
    # Early stop knobs (used only when n_trials is None)
    warmup_trials: int = 50,
    wait_trials: int = 10,
    improve_pct: float = 0.05,  # **0.05%** improvement required (NOT 5%). Pass 5 for 5%.
    # Optuna study creation
    study_name: Optional[str] = None,
    storage: Optional[str] = None,   # e.g. "sqlite:///mlp_optuna.db"
    direction: str = "minimize",          # "minimize" or "maximize"
    sampler_seed: Optional[int] = 1234,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    # Objective kwargs passthrough
    objective_kwargs: Optional[Dict[str, Any]] = None,
    # Optional dataset preview (pass splits here to be validated/printed)
    X_train: Optional[pd.DataFrame] = None,
    X_val:   Optional[pd.DataFrame] = None,
    X_test:  Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    y_val:   Optional[pd.DataFrame] = None,
    y_test:  Optional[pd.DataFrame] = None,
    condition_col: str = "type",
) -> Tuple[optuna.study.Study, Dict[str, Any], float]:
    """
    Run an Optuna study. Returns (study, best_params, best_value).

    If n_trials is None:
      • run warmup_trials unconditionally
      • then continue until no single trial achieves ≥ improve_pct (percent) improvement
        over current best during a window of wait_trials trials.
      • Example: improve_pct=0.05 means 0.05% improvement required (i.e., 0.0005 relative).
    """
    # Preview / validate splits at study START
    base_cfg = _load_yaml(base_config_path)
    _preview_and_validate_splits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        condition_col=condition_col, base_cfg=base_cfg,
    )

    # Device message (MPS note)
    req_dev = (objective_kwargs or {}).get("device", "cuda")
    dev = resolve_device(req_dev)
    print(f"[optuna] Using device: {dev}")
    if dev == "mps":
        print("[optuna] ℹ️ MPS detected — model coerces to float32 internally (your MLP already handles this).")

    print(f"[optuna] Target metric: {metric_path}")
    print(f"[optuna] Base config: {base_config_path}\n")

    # Study
    if pruner is None:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    if storage:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction=direction,
            load_if_exists=True, sampler=sampler, pruner=pruner
        )
    else:
        study = optuna.create_study(
            study_name=study_name, direction=direction, sampler=sampler, pruner=pruner
        )

    # Build the objective kwargs (inject splits + required args)
    objective_kwargs = dict(objective_kwargs or {})
    objective_kwargs.setdefault("base_config_path", base_config_path)
    objective_kwargs.setdefault("metric_path", metric_path)
    objective_kwargs.setdefault("condition_col", condition_col)
    # device is resolved inside objective again, but pass the user choice here
    objective_kwargs.setdefault("device", req_dev)

    # forward the splits (so both preview and objective use the same ones)
    objective_kwargs.setdefault("X_train", X_train)
    objective_kwargs.setdefault("X_val",   X_val)
    objective_kwargs.setdefault("X_test",  X_test)
    objective_kwargs.setdefault("y_train", y_train)
    objective_kwargs.setdefault("y_val",   y_val)
    objective_kwargs.setdefault("y_test",  y_test)

    # Fixed trials mode
    if n_trials is not None:
        study.optimize(lambda t: objective_mlp(t, **objective_kwargs), n_trials=int(n_trials))
        best = study.best_trial
        print(f"\n[optuna] ✅ Finished {n_trials} trials. Best value={best.value:.6f} (trial #{best.number}).")
        return study, best.params, float(best.value)

    # Improvement-based mode
    frac = float(improve_pct) / 100.0  # convert percent to fraction
    if frac < 0:
        raise ValueError("improve_pct must be ≥ 0 (percent).")

    # warmup
    n_done = 0
    if warmup_trials > 0:
        print(f"[optuna] Warmup: running {warmup_trials} initial trials (no early stop).")
        study.optimize(lambda t: objective_mlp(t, **objective_kwargs), n_trials=int(warmup_trials))
        n_done += warmup_trials
        print(f"[optuna] Warmup best={study.best_value:.6f} after {n_done} trials.")

    # rolling window without sufficient improvement
    no_imp = 0
    reference_best = study.best_value if len(study.trials) > 0 and study.best_value is not None else np.inf
    print(f"[optuna] Early-stop mode: require ≥ {improve_pct}% improvement within next {wait_trials} trials.")

    while True:
        prev_best = reference_best
        study.optimize(lambda t: objective_mlp(t, **objective_kwargs), n_trials=1)
        n_done += 1
        current_best = study.best_value

        # compute relative improvement (direction-aware, default 'minimize')
        if direction == "minimize":
            improved = (prev_best - current_best) / max(1e-12, prev_best)
        else:
            improved = (current_best - prev_best) / max(1e-12, abs(prev_best))

        if improved >= frac:
            reference_best = current_best
            no_imp = 0
            print(f"[optuna] 🎯 Trial #{study.best_trial.number} improved best to {current_best:.6f} "
                  f"(Δ={improved*100:.4f}%).")
        else:
            no_imp += 1
            print(f"[optuna] … no ≥{improve_pct}% improvement (Δ={improved*100:.4f}%)."
                  f" Window {no_imp}/{wait_trials}. Best still {reference_best:.6f}.")

        if no_imp >= wait_trials:
            print(f"\n[optuna] ⏹️ Stopping: no ≥{improve_pct}% improvement after {wait_trials} consecutive trials.")
            break

    best = study.best_trial
    print(f"[optuna] ✅ Finished with best value={best.value:.6f} (trial #{best.number}), total trials={n_done}.")
    return study, best.params, float(best.value)

def run_mlp_study_pipeline(
    *,
    scaled_set_name: str,
    base_config_path: str | Path,
    metric_path: str = "overall.rmse",
    # Either set n_trials (fixed) OR None to use improvement-based early stop
    n_trials: Optional[int] = 50,
    # Early-stop knobs (used only when n_trials is None)
    warmup_trials: int = 50,
    wait_trials: int = 10,
    improve_pct: float = 0.05,        # **0.05%** improvement required
    # Optuna study creation
    study_name: Optional[str] = None,
    storage: Optional[str] = None,    # e.g. "sqlite:///mlp_optuna.db"
    direction: str = "minimize",
    sampler_seed: Optional[int] = 1234,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    # MLP / data knobs
    condition_col: str = "type",
    # Anything else you want to pass straight to the objective (e.g. device="mps")
    objective_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[optuna.study.Study, Dict[str, Any], float]:
    """
    One-call pipeline:
      1) Load X/y train/val/test from data/sets/scaled_sets/<scaled_set_name>.
      2) Run Optuna study with `run_mlp_study`, printing dataset previews & shapes.

    Returns (study, best_params, best_value).
    """
    # ── 0) Sanity: dataset exists?
    available = set(list_scaled_sets())
    if scaled_set_name not in available:
        raise FileNotFoundError(
            f"Scaled set '{scaled_set_name}' not found under data/sets/scaled_sets/. "
            f"Available: {sorted(available)}"
        )

    print(f"[pipeline] Loading scaled set: {scaled_set_name}")
    X_train, X_val, X_test, y_train, y_val, y_test = load_scaled_sets(
        scaled_set_name,
        require_condition_col=condition_col,
        verbose=True,
    )

    # ── 1) Build default objective kwargs and merge user overrides
    obj_kwargs: Dict[str, Any] = {
        # Optional: give runs a recognizable prefix in logs/outputs
        "base_name_prefix": f"mlp_optuna__{scaled_set_name}",
        # Pass device if you want (e.g., "cuda" | "cpu" | "mps"); can be overridden by caller
        # "device": "cuda",
    }
    if objective_kwargs:
        obj_kwargs.update(objective_kwargs)

    # ── 2) Call your study runner (this will preview/validate splits again at start)
    study, best_params, best_value = run_mlp_study(
        base_config_path=base_config_path,
        metric_path=metric_path,
        n_trials=n_trials,
        warmup_trials=warmup_trials,
        wait_trials=wait_trials,
        improve_pct=improve_pct,
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler_seed=sampler_seed,
        pruner=pruner,
        objective_kwargs=obj_kwargs,
        # forward the loaded splits
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        condition_col=condition_col,
    )

    return study, best_params, best_value

# ──────────────────────────────────────────────────────────────────────────────
# Trials → DataFrame (handy posthoc)
# ──────────────────────────────────────────────────────────────────────────────
def trials_to_df(study: optuna.study.Study) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        row = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "datetime_start": t.datetime_start,
            "datetime_complete": t.datetime_complete,
            "duration_s": (t.duration.total_seconds() if t.duration else None),
        }
        for k, v in (t.params or {}).items():
            row[f"param.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)

def make_storage_path(study_name: str) -> str:
    """
    Returns a SQLite storage URL under:
        ROOT_PATH/outputs/mlp/studies/<study_name>/study.db
    """
    base = Path(ROOT_PATH) / "outputs" / "models"/ "mlp" / "studies" / study_name
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / 'study.db'}"