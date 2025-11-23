# train_mlp.py
import os, math, json, shutil, random, secrets
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import ContextMLPRegressor
from training.utils import (
    load_yaml_config,
    ROOT_PATH,
    flowpre_log,
    setup_training_logs_and_dirs,
    log_epoch_diagnostics,
)
from data.sets import load_or_create_raw_splits

from losses.mlp_loss import mlp_loss

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

def compute_feature_influence_mlp(
    model: nn.Module,
    x: torch.Tensor,
    c: torch.Tensor,
    feature_names: List[str],
    target_names: Optional[List[str]] = None,
    influence_step_fraction: float = 0.01,
    sweep_range: Tuple[float, float] = (-3.0, 3.0),
) -> dict:
    """
    Sweep each INPUT feature x_j over a range and measure variability in ŷ.

    Assumptions:
      - Inputs are on a comparable scale (e.g., standardized); otherwise adjust sweep_range.
      - For context, we fix a single class label (using the first label in c).

    Returns:
      {
        "feature_name_0": {"y_0": (raw_std, norm), "y_1": (...), ...},
        "feature_name_1": {...},
        ...
      }
      where 'raw_std' is the std of predictions across the sweep for that target,
      and 'norm' is that std normalized across features for the same target.
    """
    model.eval()
    device = getattr(model, "device", next(model.parameters()).device)

    # infer y-dim (and move once)
    with torch.no_grad():
        y_probe = model(x[:2].to(device), c[:2].to(device))
    y_dim = int(y_probe.shape[1])
    if target_names is None:
        target_names = [f"y_{k}" for k in range(y_dim)]

    # ensure feature_names matches x-dim
    if not feature_names or len(feature_names) != x.shape[1]:
        feature_names = [f"x_{j}" for j in range(x.shape[1])]

    # base point = mean of X (on CPU), then we sweep one coordinate at a time
    base_x = x.detach().cpu().mean(dim=0)  # (Dx,)
    sweep_min, sweep_max = float(sweep_range[0]), float(sweep_range[1])
    steps = max(2, int(round(1.0 / max(influence_step_fraction, 1e-4))))
    sweep_vals = torch.linspace(sweep_min, sweep_max, steps=steps)

    # use a single fixed class label (same pattern as FlowPre/FlowGen helpers)
    c_ref = c[:1].detach().cpu().repeat(steps)

    # collect raw std per feature per target
    raw_std_matrix = np.zeros((x.shape[1], y_dim), dtype=np.float64)

    with torch.no_grad():
        for j in range(x.shape[1]):
            x_sweep = base_x.repeat(steps, 1)  # (S, Dx)
            x_sweep[:, j] = sweep_vals  # absolute sweep; assume standardized inputs

            y_pred = model(x_sweep.to(device), c_ref.to(device))  # (S, Dy)
            y_std = torch.std(y_pred, dim=0).detach().cpu().numpy()  # (Dy,)
            raw_std_matrix[j, :] = y_std

    # normalize per target across features
    totals = raw_std_matrix.sum(axis=0)  # (Dy,)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_matrix = np.divide(
            raw_std_matrix, totals[None, :],
            out=np.zeros_like(raw_std_matrix), where=totals[None, :] > 0
        )

    # build nested dict
    influence = {}
    for j, fname in enumerate(feature_names):
        per_tgt = {}
        for k, tname in enumerate(target_names):
            per_tgt[tname] = (
                float(np.round(raw_std_matrix[j, k], 6)),
                float(np.round(norm_matrix[j, k], 6)),
            )
        influence[fname] = per_tgt

    return influence



# ───────────────────────────────────────────────────────────────────────────────
# Determinism helper (same policy you use for FlowGen)
# ───────────────────────────────────────────────────────────────────────────────
def _maybe_set_seed(seed: int | None) -> int:
    if seed is None:
        seed = secrets.randbits(64) % (2**63 - 1)
    random.seed(seed)
    np.random.seed(int(seed % (2**32 - 1)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return int(seed)


# ───────────────────────────────────────────────────────────────────────────────
# Column filtering (preserves order; returns features then y at the end)
# ───────────────────────────────────────────────────────────────────────────────
def filter_mlp_columns(
    df: pd.DataFrame,
    cols_to_exclude: List[str],
    condition_col: str,
    y_cols: List[str] | str,
) -> pd.DataFrame:
    """
    Removes specified columns while preserving original order, and reorders as:
      [condition_col] + feature_cols + y_cols
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if condition_col not in df.columns:
        raise ValueError(f"Condition column '{condition_col}' not found in DataFrame.")
    for y in y_cols:
        if y not in df.columns:
            raise ValueError(f"Target column '{y}' not found in DataFrame.")

    excluded = set(cols_to_exclude or [])
    if condition_col in excluded:
        raise ValueError(f"Condition column '{condition_col}' cannot be excluded.")
    excluded = excluded.difference(set(y_cols))

    feature_cols = [c for c in df.columns if c not in excluded and c != condition_col and c not in y_cols]
    ordered = [condition_col] + feature_cols + list(y_cols)
    return df[ordered].copy()


# ───────────────────────────────────────────────────────────────────────────────
# Dataloader (strong schema checks + returns context)
# ───────────────────────────────────────────────────────────────────────────────
def prepare_mlp_dataloader(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    condition_col: str,
    batch_size: int,
    device: torch.device,
    *,
    df_test: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    y_cols: Optional[Union[str, List[str]]] = "init",
) -> tuple:
    """
    Returns (order matches FlowGen helper for familiarity):

      x_train, y_train,
      x_val,   y_val,
      x_test,  y_test,
      c_train, c_val, c_test,
      feature_names_x, target_names,
      train_dataset, train_dataloader
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    target_names = list(y_cols)

    def _check(df: pd.DataFrame, name: str):
        if condition_col not in df.columns:
            raise ValueError(f"[{name}] missing condition_col '{condition_col}'.")
        for yc in target_names:
            if yc not in df.columns:
                raise ValueError(f"[{name}] missing target column '{yc}'.")

    _check(df_train, "train")
    _check(df_val, "val")
    if df_test is not None:
        _check(df_test, "test")

    excluded = {"post_cleaning_index", condition_col, *target_names}
    feature_names_x = [c for c in df_train.columns if c not in excluded]

    def _features(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in excluded]

    if feature_names_x != _features(df_val):
        raise ValueError("Feature columns mismatch between train and val — names or order differ.")
    if df_test is not None and feature_names_x != _features(df_test):
        raise ValueError("Feature columns mismatch between train and test — names or order differ.")

    def _to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor(df[feature_names_x].values, dtype=torch.float32, device=device)
        y = torch.tensor(df[target_names].values,    dtype=torch.float32, device=device)
        c = torch.tensor(df[condition_col].values,   dtype=torch.long,    device=device)
        return x, y, c

    x_train, y_train, c_train = _to_tensors(df_train)
    x_val,   y_val,   c_val   = _to_tensors(df_val)
    x_test = y_test = c_test = None
    if df_test is not None:
        x_test, y_test, c_test = _to_tensors(df_test)

    # deterministic loader
    train_dataset = TensorDataset(x_train, y_train, c_train)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed if seed is not None else torch.initial_seed()))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)

    return (
        x_train, y_train,
        x_val,   y_val,
        x_test,  y_test,
        c_train, c_val, c_test,
        feature_names_x, target_names,
        train_dataset, train_loader
    )


# ───────────────────────────────────────────────────────────────────────────────
# Build ContextMLPRegressor from config
# ───────────────────────────────────────────────────────────────────────────────
def build_mlp_model(
    model_cfg: dict,
    *,
    x_dim: int,
    y_dim: int,
    num_classes: int,
    device: torch.device,
) -> ContextMLPRegressor:
    return ContextMLPRegressor(
        input_dim=x_dim,
        y_dim=y_dim,
        num_classes=num_classes,
        embedding_dim=int(model_cfg.get("embedding_dim", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        activation=str(model_cfg.get("activation", "relu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        batchnorm=bool(model_cfg.get("batchnorm", False)),
        use_weight_norm=bool(model_cfg.get("use_weight_norm", True)),
        residual=bool(model_cfg.get("residual", True)),
        final_activation=model_cfg.get("final_activation", None),
        task=str(model_cfg.get("task", "regression")),
        context_mode=str(model_cfg.get("context_mode", "embed")),
        device=device,
        dtype=torch.float32,
        feature_names_in=model_cfg.get("feature_names_in", None),
        target_names=model_cfg.get("target_names", None),
        class_names=model_cfg.get("class_names", None),
    )

# ───────────────────────────────────────────────────────────────────────────────
# SHAP (optional). Safe to call even if `shap` is missing.
# ───────────────────────────────────────────────────────────────────────────────
def compute_shap_values_for_mlp(
    model: ContextMLPRegressor,
    *,
    X_background: torch.Tensor,
    X_sample: torch.Tensor,
    class_id: int,
    nsamples: int = 200,
) -> Optional[np.ndarray]:
    """
    Returns np.ndarray of shape [X_sample.size(0), x_dim] with SHAP values
    for the *regression* head within a fixed context = class_id.
    If shap is not installed, returns None.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    model.eval()
    device = model.device

    def f(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = model.predict_with_fixed_context(x, fixed_class_id=class_id)
        return y.detach().cpu().numpy()

    bg = X_background.detach().cpu().numpy()
    xs = X_sample.detach().cpu().numpy()

    # KernelExplainer is model-agnostic and works with our wrapper f(X)
    expl = shap.KernelExplainer(f, bg)
    shap_vals = expl.shap_values(xs, nsamples=nsamples)
    # If y_dim > 1, shap returns list; take first head by default
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return np.array(shap_vals)


# ───────────────────────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────────────────────
def train_mlp_model(
    cXy_train: pd.DataFrame,
    cXy_val: pd.DataFrame,
    condition_col: str,
    *,
    cXy_test: pd.DataFrame | None = None,
    seed: int | None = None,
    config_filename: str = "mlp.yaml",
    base_name: str = "mlp",
    device: str = "cuda",
    verbose: bool = True,
):
    """
    Trains ContextMLPRegressor on (X, y) with explicit context c.

    Config (mlp.yaml) expected sections:
      model:
        embedding_dim, hidden_dim, num_layers, activation, dropout, ...
        task: ["regression"|"classification"]
        context_mode: ["embed"|"onehot"]
      training:
        learning_rate, batch_size, num_epochs,
        early_stopping_patience, lr_decay_patience, lr_decay_factor,
        min_improvement, min_improvement_floor, lr_patience_factor,
        loss_reduction: ["overall"|"per_class_equal"|"per_class_weighted"]
        save_model, save_states, save_results, log_training
      interpretability (optional):
        compute_shap: bool
        shap_num_background: int
        shap_num_samples: int
        shap_classes: list[int] (defaults to all)
    """
    # -------------------- Setup & config --------------------
    seed = _maybe_set_seed(seed)
    config = load_yaml_config(config_filename)
    model_cfg = config["model"]
    train_cfg = config["training"]
    interp_cfg = config.get("interpretability", {})

    dev = torch.device("cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    # -------------------- Logging dirs --------------------
    versioned_dir, versioned_name, log_file_path, snapshots_dir = setup_training_logs_and_dirs(
        base_name, config_filename, config, verbose, train_cfg.get("save_states", False),
        train_cfg.get("log_training", True), subdir="mlp",
    )

    # -------------------- Data --------------------
    # pick targets from the tail of cXy_val if not explicitly provided
    if "target_names" in model_cfg and model_cfg["target_names"]:
        y_cols_cfg = model_cfg["target_names"]
    else:
        # assume everything after condition_col is X then the last column is y
        non_meta = [c for c in cXy_val.columns if c not in ["post_cleaning_index", condition_col]]
        y_cols_cfg = [non_meta[-1]]  # last column as target

    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        c_train, c_val, c_test,
        feature_names_x, target_names,
        train_dataset, train_loader
    ) = prepare_mlp_dataloader(
        df_train=cXy_train,
        df_val=cXy_val,
        condition_col=condition_col,
        batch_size=int(train_cfg["batch_size"]),
        device=dev,
        df_test=cXy_test,
        seed=seed,
        y_cols=y_cols_cfg,
    )

    # Build model
    model = build_mlp_model(
        model_cfg=model_cfg,
        x_dim=x_train.shape[1],
        y_dim=y_train.shape[1],
        num_classes=int(c_train.max().item()) + 1,
        device=dev,
    )

    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    betas = tuple(train_cfg.get("betas", [0.9, 0.999]))
    eps = float(train_cfg.get("eps", 1e-8))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    lr = float(train_cfg["learning_rate"])

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Use 'adam' or 'adamw'.")

    reduction_mode = str(train_cfg.get("loss_reduction", "overall"))
    task = str(model_cfg.get("task", "regression"))

    reg_group_metric = str(train_cfg.get("regression_group_metric", "mse")).lower()
    if reg_group_metric not in {"mse", "rmse", "rrmse"}:
        raise ValueError(f"Unsupported regression_group_metric '{reg_group_metric}'. Use 'mse' | 'rmse' | 'rrmse'.")

    # -------------------- Train loop --------------------
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = None

    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 30))
    total_epochs = int(train_cfg["num_epochs"])
    log_training = bool(train_cfg.get("log_training", True))
    epochs_no_improve = 0

    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))

    sched_name = str(train_cfg.get("lr_scheduler", "plateau")).lower()
    scheduler = None
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_decay_factor", 0.5)),
            patience=int(train_cfg.get("lr_decay_patience", 12)),
            threshold=float(train_cfg.get("plateau_threshold", 1e-4)),
            threshold_mode=str(train_cfg.get("plateau_threshold_mode", "rel")),
            cooldown=int(train_cfg.get("lr_cooldown", 0)),
            min_lr=float(train_cfg.get("min_lr", 0.0)),

        )

    flowpre_log(f"Using device: {dev}", filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🎲 Using seed: {seed}", filename_or_path=log_file_path, verbose=verbose)

    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0
        diag_accum = {}

        for batch_idx, (bx, by, bc) in enumerate(train_loader, start=1):
            bx, by, bc = bx.to(dev), by.to(dev), bc.to(dev)
            loss, diag = mlp_loss(
                model, bx, by, bc,
                task=task,
                reduction_mode=reduction_mode,
                regression_group_metric=reg_group_metric,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * bx.size(0)
            for k, v in diag.items():
                diag_accum.setdefault(k, []).append(float(v))

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation (full-batch)
        model.eval()
        with torch.no_grad():
            val_loss, val_diag = mlp_loss(
                model, x_val, y_val, c_val,
                task=task,
                reduction_mode=reduction_mode,
                regression_group_metric=reg_group_metric,
            )

        log_epoch_diagnostics(epoch, diag_accum, log_file_path, verbose)

        flowpre_log(
            f"📉 Epoch {epoch + 1}/{total_epochs} — Train Loss: {avg_train_loss:.6f} | Val Loss: {float(val_loss):.6f}",
            filename_or_path=log_file_path, verbose=verbose
        )

        # keep best (with explicit min-delta)
        improved = (best_val - float(val_loss)) > early_stop_min_delta
        if improved:
            best_val = float(val_loss)
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            if train_cfg.get("save_states", False):
                snap = snapshots_dir / f"{versioned_name}_epoch{epoch + 1}_valloss{float(val_loss):.6f}.pt"
                torch.save(best_state, snap)
                flowpre_log(f"💾 Saved snapshot: {snap.name}", filename_or_path=log_file_path, verbose=verbose)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # step the scheduler (if enabled)
        if scheduler is not None:
            scheduler.step(float(val_loss))

        # early stopping
        if epochs_no_improve >= early_stopping_patience:
            flowpre_log(f"🛌 Early stopping at epoch {epoch + 1}",
                        filename_or_path=log_file_path, verbose=verbose)
            break

        flowpre_log(
            f"🔎 BestVal: {best_val:.6f} | Δ={best_val - float(val_loss):+.6f} | "
            f"Patience {epochs_no_improve}/{early_stopping_patience} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}",
            filename_or_path=log_file_path, verbose=verbose
        )

    # Load best
    model.load_state_dict(best_state)

    # ── Final metrics (overall + per-class) on train/val/(test)
    def _metrics_split(x, y, c) -> dict:
        model.eval()
        with torch.no_grad():
            y_hat = model(x, c)

            err = y_hat - y
            mse = float(torch.mean(err ** 2))
            rmse = float(torch.sqrt(torch.tensor(mse) + 1e-12))
            rrmse = float(
                (torch.sqrt(torch.mean(err ** 2)) /
                 (torch.sqrt(torch.mean(y ** 2)) + 1e-12)).item()
            )
            mae = float(torch.mean(torch.abs(err)))
            mape = float(torch.mean(torch.abs(err) / (torch.clamp(torch.abs(y), min=1e-8))).item())

            y_mu = y.mean(dim=0)
            ss_res = torch.sum((y - y_hat) ** 2)
            ss_tot = torch.sum((y - y_mu) ** 2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

            # per-class
            out_pc = {}
            for cls in torch.unique(c):
                m = (c == cls)
                if not torch.any(m):
                    continue
                yh = y_hat[m];
                yt = y[m]
                e = yh - yt
                mse_c = float(torch.mean(e ** 2))
                rmse_c = float(torch.sqrt(torch.mean(e ** 2) + 1e-12))
                rrmse_c = float(
                    (torch.sqrt(torch.mean(e ** 2)) /
                     (torch.sqrt(torch.mean(yt ** 2)) + 1e-12)).item()
                )
                mae_c = float(torch.mean(torch.abs(e)))
                mape_c = float(torch.mean(torch.abs(e) / (torch.clamp(torch.abs(yt), min=1e-8))).item())

                yt_mu = yt.mean(dim=0)
                ss_res_c = torch.sum((yt - yh) ** 2)
                ss_tot_c = torch.sum((yt - yt_mu) ** 2)
                r2_c = float(1.0 - ss_res_c / (ss_tot_c + 1e-12))

                out_pc[int(cls.item())] = {
                    "mse": mse_c, "rmse": rmse_c, "rrmse": rrmse_c,
                    "mae": mae_c, "mape": mape_c, "r2": r2_c,
                    "n": int(m.sum().item()),
                }

        overall = {"mse": mse, "rmse": rmse, "rrmse": rrmse, "mae": mae, "mape": mape, "r2": r2}
        return {"overall": overall, "per_class": out_pc}

    train_metrics = _metrics_split(x_train, y_train, c_train)
    val_metrics   = _metrics_split(x_val, y_val, c_val)
    test_metrics  = _metrics_split(x_test, y_test, c_test) if (x_test is not None) else None

    tm, vm = train_metrics["overall"], val_metrics["overall"]
    flowpre_log(
        f"📈 Train — R²: {tm['r2']:.4f} | MAE: {tm['mae']:.6f} | MAPE: {tm['mape']:.4f} | "
        f"MSE: {tm['mse']:.6f} | RMSE: {tm['rmse']:.6f} | RRMSE: {tm['rrmse']:.6f}",
        filename_or_path=log_file_path, verbose=verbose
    )
    flowpre_log(
        f"📈 Val   — R²: {vm['r2']:.4f} | MAE: {vm['mae']:.6f} | MAPE: {vm['mape']:.4f} | "
        f"MSE: {vm['mse']:.6f} | RMSE: {vm['rmse']:.6f} | RRMSE: {vm['rrmse']:.6f}",
        filename_or_path=log_file_path, verbose=verbose
    )

    # ── Optional SHAP (class-conditional)
    shap_results = None
    if bool(interp_cfg.get("compute_shap", False)):
        bg_n = int(interp_cfg.get("shap_num_background", min(128, x_train.shape[0])))
        sm_n = int(interp_cfg.get("shap_num_samples", min(256, x_val.shape[0])))
        cls_list = interp_cfg.get("shap_classes", None)
        if cls_list is None:
            cls_list = [int(i) for i in torch.unique(c_train).tolist()]

        # simple uniform samples (no replace if possible)
        bg_idx = torch.randperm(x_train.shape[0])[:bg_n]
        sm_idx = torch.randperm(x_val.shape[0])[:sm_n]
        X_bg = x_train[bg_idx]
        X_sm = x_val[sm_idx]

        shap_results = {}
        for cls in cls_list:
            sv = compute_shap_values_for_mlp(model, X_background=X_bg, X_sample=X_sm, class_id=int(cls),
                                             nsamples=int(interp_cfg.get("shap_kernel_nsamples", 200)))
            if sv is not None:
                # aggregate |SHAP| mean per feature
                shap_results[int(cls)] = {
                    "mean_abs": {name: float(np.mean(np.abs(sv[:, j]))) for j, name in enumerate(feature_names_x)}
                }
            else:
                shap_results = None
                flowpre_log("ℹ️ SHAP not installed; skipping SHAP computation.",
                            filename_or_path=log_file_path, verbose=verbose)
                break
    # --- Optional feature influence (input sweep; MLP has no latent/inverse)
    influence_dict = None
    if bool(interp_cfg.get("save_influence", False)):
        influence_dict = compute_feature_influence_mlp(
            model,
            x_train, c_train,
            feature_names=feature_names_x,
            target_names=target_names,
            influence_step_fraction=float(interp_cfg.get("influence_step_fraction", 0.01)),
            sweep_range=tuple(interp_cfg.get("sweep_range", (-3.0, 3.0))),
        )
        with open(versioned_dir / f"{versioned_name}_influence.json", "w") as f:
            json.dump(influence_dict, f, indent=2)
        flowpre_log(
            f"🧠 Influence saved: {versioned_name}_influence.json",
            filename_or_path=log_file_path, verbose=verbose
        )

    # ── Save results
    if bool(train_cfg.get("save_results", False)):
        results = {
            "seed": seed,
            "best_epoch": best_epoch,
            "val_best_loss": round(best_val, 6),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "loss_reduction": reduction_mode,
            "regression_group_metric": reg_group_metric,
            "task": task,
            "feature_names_x": feature_names_x,
            "target_names": target_names,
            "shap": shap_results,
        }
        results_path = versioned_dir / f"{versioned_name}_results.yaml"
        import yaml
        with open(results_path, "w") as f:
            yaml.dump(results, f, sort_keys=False)
        flowpre_log(f"📝 Saved results to: {results_path}", filename_or_path=log_file_path, verbose=verbose)

    # Save model(s)
    if bool(train_cfg.get("save_model", False)):
        final_path = versioned_dir / f"{versioned_name}.pt"
        torch.save(best_state, final_path)
        flowpre_log(f"💾 Saved model: {final_path.name}", filename_or_path=log_file_path, verbose=verbose)

    # Always save the config for reproducibility
    shutil.copy(ROOT_PATH / "config" / config_filename, versioned_dir / f"{versioned_name}.yaml")
    flowpre_log(f"✅ Config saved under {versioned_dir}", filename_or_path=log_file_path, verbose=verbose)

    return (model, val_metrics)


# ───────────────────────────────────────────────────────────────────────────────
# Convenience pipeline (loads your existing splits)
# ───────────────────────────────────────────────────────────────────────────────
def train_mlp_pipeline(
    condition_col: str = "type",
    config_filename: str = "mlp.yaml",
    base_name: str = "mlp",
    device: str = "cuda",
    seed: int | None = None,
    verbose: bool = True,
    *,
    # Optional external splits
    X_train: pd.DataFrame | None = None,
    X_val:   pd.DataFrame | None = None,
    X_test:  pd.DataFrame | None = None,
    y_train: pd.DataFrame | None = None,
    y_val:   pd.DataFrame | None = None,
    y_test:  pd.DataFrame | None = None,
):
    """
    If all three X and all three y splits are provided, use them.
    Otherwise, load the default splits from data.sets.
    Merges into cXy frames ordered as: [post_cleaning_index, condition_col, X..., y...].
    """

    have_all_user_splits = all(
        df is not None for df in (X_train, X_val, X_test, y_train, y_val, y_test)
    )

    if not have_all_user_splits:
        # fallback to your loader (r_* not needed anymore)
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         _r_train, _r_val, _r_test) = load_or_create_raw_splits(condition_col=condition_col, verbose=verbose)

    def _build_cXy(X_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        X_df = X_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
        y_df = y_df.sort_values("post_cleaning_index").reset_index(drop=True).copy()
        df = pd.merge(X_df, y_df, on="post_cleaning_index", how="inner", validate="one_to_one")
        x_cols = [c for c in X_df.columns if c not in ("post_cleaning_index", condition_col)]
        y_cols = [c for c in y_df.columns if c != "post_cleaning_index"]
        return df[["post_cleaning_index", condition_col] + x_cols + y_cols].copy()

    cXy_tr = _build_cXy(X_train, y_train)
    cXy_va = _build_cXy(X_val,   y_val)
    cXy_te = _build_cXy(X_test,  y_test)

    return train_mlp_model(
        cXy_train=cXy_tr,
        cXy_val=cXy_va,
        cXy_test=cXy_te,
        condition_col=condition_col,
        config_filename=config_filename,
        base_name=base_name,
        device=device,
        seed=seed,
        verbose=verbose,
    )

