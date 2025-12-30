# training_scripts/mlp_scripts/run_all_scaled_sets_bestonly.py
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

import optuna
import yaml

# MINIMAL CHANGE: swap scaled -> augmented utils
from training.utils import (
    ROOT_PATH,
    list_augmented_scaled_sets,
    load_augmented_scaled_sets,
)

from training.train_mlp import train_mlp_pipeline
from training.optuna_mlp import (
    resolve_device,
    suggest_mlp_params,
    build_trial_config,
    make_storage_path,
    compute_real_scale_objective_loss,
)


MLP_OUT_DIR = Path(ROOT_PATH) / "outputs" / "models" / "mlp"
CONFIG_DIR = Path(ROOT_PATH) / "config"


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _write_yaml(d: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def _list_run_dirs() -> Dict[str, Path]:
    if not MLP_OUT_DIR.exists():
        return {}
    out = {}
    for p in MLP_OUT_DIR.iterdir():
        if p.is_dir():
            out[p.name] = p
    return out


def _find_new_run_dir(before: Dict[str, Path], after: Dict[str, Path], base_name: str) -> Optional[Path]:
    # Prefer truly-new dirs; then fall back to "starts with base_name" if something already existed.
    new_names = [k for k in after.keys() if k not in before]
    new_dirs = [after[n] for n in new_names if after[n].is_dir()]

    # Filter to those that look like they belong to this trial
    candidates = [p for p in new_dirs if p.name.startswith(base_name)]
    if not candidates:
        # fallback: any dir starting with base_name, pick most recently modified
        candidates = [p for p in after.values() if p.is_dir() and p.name.startswith(base_name)]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _cleanup_paths(paths: List[Path]) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            # best-effort cleanup
            pass


def _move_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))


def run_one_scaled_set(
    *,
    scaled_set_name: str,
    base_config_path: Path,
    metric_path: str,
    condition_col: str,
    device_req: str,
    sampler_seed: int,
    objective_seed: Optional[int],
    seed_base: Optional[int],
    num_epochs_override: Optional[int],
    # trials policy
    n_trials: Optional[int],
    warmup_trials: int,
    wait_trials: int,
    improve_pct: float,
    timeout_s: Optional[int],
    verbose: bool,
    keep_topk: int,
) -> Tuple[bool, Optional[float], Optional[Path]]:
    """
    Returns:
      (ok, best_value, best_dir)
    """

    # Load dataset splits once
    print(f"[dataset] Loading scaled set: {scaled_set_name}")
    # MINIMAL CHANGE: load augmented splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_augmented_scaled_sets(
        scaled_set_name,
        require_condition_col=condition_col,
        verbose=True,
    )

    # MINIMAL CHANGE: y_scaler comes from augmented_scaled_sets/<name>/scalers
    y_scaler_path = (
        Path(ROOT_PATH)
        / "data"
        / "sets"
        / "augmented_scaled_sets"
        / scaled_set_name
        / "scalers"
        / f"{scaled_set_name}_y_scaler.pkl"
    )

    if y_scaler_path.exists():
        y_scaler = joblib.load(y_scaler_path)
    else:
        # fallback (in case filename differs): find ONE pkl that looks like y scaler
        y_scaler_dir = y_scaler_path.parent
        candidates = []
        if y_scaler_dir.exists():
            candidates.extend(sorted(y_scaler_dir.glob("*_y_scaler.pkl")))
            if not candidates:
                candidates.extend(sorted(y_scaler_dir.glob("*y*scaler*.pkl")))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"[{scaled_set_name}] y scaler not found at expected path:\n"
                f"  {y_scaler_path}\n"
                f"Fallback search in:\n"
                f"  {y_scaler_dir}\n"
                f"Candidates found:\n"
                f"  {candidates}"
            )
        y_scaler = joblib.load(candidates[0])


    safe_metric = metric_path.replace(".", "_")
    study_name = _safe_name(f"mlp_{safe_metric}_{scaled_set_name}")
    storage = make_storage_path(study_name)

    print("=" * 110)
    print(f"▶️  STUDY: {scaled_set_name}")
    print(f"    study_name  : {study_name}")
    print(f"    metric_path : {metric_path}")
    print(f"    storage     : {storage}")
    print(f"    timeout_s   : {timeout_s}")
    print("=" * 110)

    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )

    dev = resolve_device(device_req)

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    loss_reduction = base_cfg.get("training", {}).get("loss_reduction", "overall")
    reg_group_metric = base_cfg.get("training", {}).get("regression_group_metric", "mse")

    print("\n" + "-" * 110)
    print("🔧 FIXED RUN CONFIG (will NOT change across trials)")
    print(f"• scaled_set             : {scaled_set_name}")
    print(f"• device                 : {dev}")
    print(f"• optuna sampler         : TPESampler")
    print(f"• sampler seed           : {sampler_seed}")
    print(f"• objective_seed         : {objective_seed}")
    print(f"• seed_base              : {seed_base}")
    print(f"• metric_path            : {metric_path}")

    print("• objective metric definition:")
    print(f"    - loss_reduction           : {loss_reduction}")
    print(f"    - regression_group_metric  : {reg_group_metric}")

    print(f"• num_epochs_override   : {num_epochs_override}")
    print(f"• warmup_trials         : {warmup_trials}")
    print(f"• wait_trials           : {wait_trials}")
    print(f"• improve_pct           : {improve_pct}%")
    print(f"• timeout_s             : {timeout_s}")
    print(f"• keep_topk             : {keep_topk}")
    print("-" * 110 + "\n")

    start_t = time.time()

    study_start_t = start_t
    trial_times: List[float] = []

    def objective(trial: optuna.trial.Trial) -> float:
        trial_start_t = time.time()

        # trial params
        trial_params = suggest_mlp_params(trial)

        # seed policy
        if objective_seed is not None:
            trial_seed = int(objective_seed)
        elif seed_base is not None:
            trial_seed = int(seed_base + trial.number)
        else:
            trial_seed = None

        # force “trial artifacts” ON so the best trial has a .pt without retraining
        # (we’ll delete non-best dirs after the study)
        extra_overrides = {
            "training.save_model": True,
            "training.save_results": True,
            "training.save_states": False,
            "training.log_training": False,
            # (optional) reduce noise; keep interpretability off
            "interpretability.compute_shap": False,
            "interpretability.save_influence": False,
        }

        # build trial config dict
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = build_trial_config(
            base_cfg=base_cfg,
            trial_params=trial_params,
            seed=trial_seed,
            num_epochs_override=num_epochs_override,
            extra_overrides=extra_overrides,
        )

        # write per-trial config into ROOT/config so train_mlp can load it
        cfg_filename = _safe_name(f"mlp__{study_name}__t{trial.number:05d}.yaml")
        cfg_path = CONFIG_DIR / cfg_filename
        _write_yaml(cfg, cfg_path)

        base_name = f"mlp_optuna__{scaled_set_name}__t{trial.number:05d}"

        # detect newly created run dir
        before = _list_run_dirs()

        # train
        model, val_metrics = train_mlp_pipeline(
            condition_col=condition_col,
            config_filename=cfg_filename,
            base_name=base_name,
            device=dev,
            seed=trial_seed,
            verbose=bool(verbose),
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )

        # optuna objective (REAL scale, same definition as training loss)
        train_cfg = cfg.get("training", {})
        value = float(
            compute_real_scale_objective_loss(
                model=model,
                X_val=X_val,
                y_val=y_val,
                condition_col=condition_col,
                y_scaler=y_scaler,
                reduction_mode=str(train_cfg.get("loss_reduction", "overall")),
                regression_group_metric=str(train_cfg.get("regression_group_metric", "mse")),
            )
        )

        after = _list_run_dirs()
        run_dir = _find_new_run_dir(before, after, base_name)

        # track artifacts for cleanup/keeping
        trial.set_user_attr("scaled_set_name", scaled_set_name)
        trial.set_user_attr("config_filename", cfg_filename)
        trial.set_user_attr("run_dir", str(run_dir) if run_dir is not None else "")
        trial.set_user_attr("device", dev)
        trial.set_user_attr("metric_path", metric_path)
        trial.set_user_attr("value", value)

        # ---- timing stats ----
        trial_elapsed = time.time() - trial_start_t
        trial_times.append(trial_elapsed)

        elapsed_study = time.time() - study_start_t
        mean_trial_time = sum(trial_times) / len(trial_times)

        print(
            f"\n⏱️  Trial {trial.number} timing\n"
            f"   time elapsed since beginning of study: {elapsed_study / 60:.2f} min\n"
            f"   mean time per trial                 : {mean_trial_time:.2f} s\n"
        )

        return value

    # ---- optimize with either fixed n_trials or improvement-based loop ----
    try:
        if n_trials is not None:
            study.optimize(objective, n_trials=int(n_trials), timeout=timeout_s)
        else:
            # improvement-based (with timeout respected)
            frac = float(improve_pct) / 100.0

            # warmup
            if warmup_trials > 0:
                study.optimize(objective, n_trials=int(warmup_trials), timeout=timeout_s)

            no_imp = 0
            reference_best = study.best_value if study.best_value is not None else float("inf")

            while True:
                if timeout_s is not None and (time.time() - start_t) >= timeout_s:
                    print("[optuna] ⏱️ timeout reached, stopping.")
                    break

                prev_best = reference_best
                study.optimize(objective, n_trials=1)

                current_best = study.best_value
                if current_best is None:
                    no_imp += 1
                else:
                    improved = (prev_best - current_best) / max(1e-12, prev_best)
                    if improved >= frac:
                        reference_best = current_best
                        no_imp = 0
                    else:
                        no_imp += 1

                if no_imp >= int(wait_trials):
                    break

    except Exception as e:
        print(f"❌ Study failed for {scaled_set_name}: {type(e).__name__}: {e}")
        return False, None, None

    if study.best_trial is None or study.best_value is None:
        print(f"❌ No best trial produced for {scaled_set_name}.")
        return False, None, None

    # ---- collect run dirs + configs for cleanup ----
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: float(t.value),
    )
    if not trials_sorted:
        print(f"❌ No completed trials for {scaled_set_name}.")
        return False, None, None

    keep_topk = max(1, int(keep_topk))
    keep_trials = trials_sorted[:keep_topk]
    keep_dirs = []
    keep_cfgs = []
    for t in keep_trials:
        rd = (t.user_attrs.get("run_dir") or "").strip()
        cf = (t.user_attrs.get("config_filename") or "").strip()
        if rd:
            keep_dirs.append(Path(rd))
        if cf:
            keep_cfgs.append(CONFIG_DIR / cf)

    # everything else: delete
    del_dirs = []
    del_cfgs = []
    for t in study.trials:
        rd = (t.user_attrs.get("run_dir") or "").strip()
        cf = (t.user_attrs.get("config_filename") or "").strip()
        if rd:
            p = Path(rd)
            if p not in keep_dirs:
                del_dirs.append(p)
        if cf:
            p = CONFIG_DIR / cf
            if p not in keep_cfgs:
                del_cfgs.append(p)

    # also delete configs even for kept trials (they’re copied into the run dir anyway)
    # (keeping config folder clean)
    del_cfgs.extend(keep_cfgs)

    # ---- move best dir into outputs/models/mlp/best/<scaled_set_name> ----
    best_trial = keep_trials[0]
    best_dir_str = (best_trial.user_attrs.get("run_dir") or "").strip()
    best_dir = Path(best_dir_str) if best_dir_str else None

    if best_dir is None or not best_dir.exists():
        print("⚠️ Could not locate best run_dir to keep. Will only cleanup configs.")
        _cleanup_paths(del_cfgs)
        return True, float(study.best_value), None

    best_target = MLP_OUT_DIR / "best" / scaled_set_name
    _move_dir(best_dir, best_target)

    # delete all other dirs + configs
    _cleanup_paths(del_dirs)
    _cleanup_paths(del_cfgs)

    # write a short summary next to the model
    summary = {
        "scaled_set_name": scaled_set_name,
        "study_name": study.study_name,
        "storage": storage,
        "metric_path": metric_path,
        "device": dev,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": dict(study.best_trial.params),
        "kept_topk": keep_topk,
        "saved_dir": str(best_target),
    }
    with open(best_target / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_study_time = time.time() - study_start_t
    print(f"\n⏱️  Total study time for {scaled_set_name}: {total_study_time / 60:.2f} minutes\n")

    print(f"✅ Kept BEST only → {best_target}")
    return True, float(study.best_value), best_target


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna MLP studies over all scaled sets, keeping ONLY the best run artifacts per set."
    )
    parser.add_argument("--exclude", type=str, nargs="*", default=[], help="Scaled set names to skip.")
    parser.add_argument("--metric-path", type=str, default="per_class.rrmse",
                        help="Metric path (e.g. 'overall.rmse', 'per_class.rrmse', 'per_class.0.rmse').")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Sampler seed for Optuna (TPE).")
    parser.add_argument("--objective-seed", type=int, default=None,
                        help="If set, forces SAME training seed for every trial.")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="If objective-seed is None, uses (seed_base + trial.number) as per-trial seed.")
    parser.add_argument("--warmup-trials", type=int, default=50,
                        help="Warmup trials (only used when --n-trials is omitted).")
    parser.add_argument("--wait-trials", type=int, default=10,
                        help="Stop after this many consecutive non-improving trials (only when --n-trials omitted).")
    parser.add_argument("--improve-pct", type=float, default=0.05,
                        help="Required relative improvement in percent (0.05 = 0.05%).")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="If set, run fixed number of trials (ignores warmup/wait early-stop).")
    parser.add_argument("--timeout-min", type=int, default=60,
                        help="Time budget per scaled set study in minutes (best-effort; stops between trials).")
    parser.add_argument("--num-epochs-override", type=int, default=120,
                        help="Force training.num_epochs for ALL trials (strongly recommended for 1h/set). "
                             "Use 0 to disable override.")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu", "mps"],
                        help="Device preference.")
    parser.add_argument("--condition-col", type=str, default="type",
                        help="Context column name.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose training logs (slower).")
    parser.add_argument("--keep-topk", type=int, default=1,
                        help="Keep top-K trial run folders instead of just best (default 1).")
    args = parser.parse_args()

    script_start_t = time.time()

    base_config_path = Path(ROOT_PATH) / "config" / "mlp.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    # MINIMAL CHANGE: list augmented sets instead of scaled sets
    all_sets = sorted(list_augmented_scaled_sets())
    run_sets = [s for s in all_sets if s not in set(args.exclude or [])]

    print("Available scaled sets:", all_sets)
    print("Excluding:", args.exclude)

    print("\n" + "=" * 110)
    print("🚀 STARTING MLP OPTUNA STUDIES")
    print(f"Total scaled sets to run: {len(run_sets)}")
    for i, s in enumerate(run_sets, 1):
        print(f"  {i:02d}. {s}")
    print("=" * 110 + "\n")

    timeout_s = None if args.timeout_min is None else int(max(0, args.timeout_min) * 60)
    num_epochs_override = None if (args.num_epochs_override is None or args.num_epochs_override <= 0) else int(args.num_epochs_override)

    results: List[Tuple[str, float, str]] = []
    failures: List[Tuple[str, str]] = []

    for scaled_set_name in run_sets:
        ok, best_val, best_dir = run_one_scaled_set(
            scaled_set_name=scaled_set_name,
            base_config_path=base_config_path,
            metric_path=args.metric_path,
            condition_col=args.condition_col,
            device_req=args.device,
            sampler_seed=args.seed,
            objective_seed=args.objective_seed,
            seed_base=args.seed_base,
            num_epochs_override=num_epochs_override,
            n_trials=args.n_trials,
            warmup_trials=args.warmup_trials,
            wait_trials=args.wait_trials,
            improve_pct=args.improve_pct,
            timeout_s=timeout_s,
            verbose=bool(args.verbose),
            keep_topk=int(args.keep_topk),
        )
        if ok and best_val is not None:
            results.append((scaled_set_name, float(best_val), str(best_dir) if best_dir else ""))
        else:
            failures.append((scaled_set_name, "study failed or produced no best"))

    print("\n" + "#" * 110)
    print("SUMMARY")
    if results:
        for name, val, d in sorted(results, key=lambda t: t[1]):
            print(f"  {name:<55} best={val:.6f}  kept={d}")
    else:
        print("  No successful runs.")

    if failures:
        print("\nFAILURES:")
        for name, msg in failures:
            print(f"  {name:<55} {msg}")
    print("#" * 110)

    total_script_time = time.time() - script_start_t
    print(f"\n⏱️  TOTAL SCRIPT TIME: {total_script_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()
