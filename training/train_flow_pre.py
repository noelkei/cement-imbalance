import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from models.flow_pre import FlowPre
from training.utils import load_yaml_config
from training.utils import ROOT_PATH
from training.utils import flowpre_log, setup_training_logs_and_dirs, log_epoch_diagnostics
import shutil
import re
import math
import json
import numpy as np
from copy import deepcopy
import yaml
from data.sets import load_or_create_raw_splits
# 🔍 Sanity check base distribution manually
from nflows.distributions.normal import StandardNormal

from losses.flow_pre_loss import flexible_flow_loss_from_model

from typing import Optional, Tuple

from scipy.stats import skew, kurtosis, chi2, kstest
import os, random, secrets

def _latent_isotropy_stats_from_z(z_np: np.ndarray) -> dict:
    """
    Diagnostics on latent matrix z (N, D):
      - skewness_mean: mean |skew| across dims (target 0)
      - kurtosis_mean: mean Pearson kurtosis across dims (target ~3)
      - mahalanobis_mean/median: on distances
      - mahalanobis_ks_p: KS p-value of d^2 vs ChiSquare(D)
      - eigval_std: std of covariance eigenvalues (lower ~ more isotropic)
    """
    N, D = z_np.shape
    sk = skew(z_np, axis=0, bias=False)                     # target ~ 0
    ku = kurtosis(z_np, axis=0, fisher=False, bias=False)   # target ~ 3 (Pearson)

    skewness_mean = float(np.mean(np.abs(sk)))
    kurtosis_mean = float(np.mean(ku))

    cov = np.cov(z_np, rowvar=False)
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)
    evals = np.linalg.eigvalsh(cov_reg)
    eigval_std = float(np.std(evals))

    mu = z_np.mean(axis=0, keepdims=True)
    w, V = np.linalg.eigh(cov_reg)
    w_inv_sqrt = np.diag(1.0 / np.sqrt(w))
    z_whitened = (z_np - mu) @ V @ w_inv_sqrt
    d2 = np.sum(z_whitened**2, axis=1)
    d = np.sqrt(d2)

    mahalanobis_mean = float(np.mean(d))
    mahalanobis_median = float(np.median(d))
    ks_stat, ks_p = kstest(d2, chi2(df=D).cdf)
    mahalanobis_ks_p = float(ks_p)

    return {
        "skewness_mean": skewness_mean,
        "kurtosis_mean": kurtosis_mean,
        "mahalanobis_mean": mahalanobis_mean,
        "mahalanobis_median": mahalanobis_median,
        "mahalanobis_ks_p": mahalanobis_ks_p,
        "eigval_std": eigval_std,
    }


def _latent_isotropy_stats(model, x: torch.Tensor, c: torch.Tensor) -> dict:
    """Use FlowPre.forward(x, c) to get z, then compute whole-set diagnostics."""
    model.eval()
    with torch.no_grad():
        z, _ = model.forward(x, c)   # ✅ matches your FlowPre API
    return _latent_isotropy_stats_from_z(z.detach().cpu().numpy())


def _latent_isotropy_stats_per_class(model, x: torch.Tensor, c: torch.Tensor) -> dict[int, dict]:
    """Per-class diagnostics with sample counts."""
    model.eval()
    out = {}
    with torch.no_grad():
        for cls in torch.unique(c):
            m = (c == cls)
            if not m.any():
                continue
            z, _ = model.forward(x[m], c[m])  # ✅ same API
            stats = _latent_isotropy_stats_from_z(z.detach().cpu().numpy())
            stats["n"] = int(m.sum().item())
            out[int(cls.item())] = {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                    for k, v in stats.items()}
    return out


# --- Helper: compute per-class isotropy RRMSE (from loss function outputs) ---
def _iso_rrmse_per_class_from_loss(model, x, c, loss_kwargs):
    """
    For each class label in c, run flexible_flow_loss_from_model on the subset and
    collect (rrmse_mean, rrmse_std) that quantify how close z_mean→0 and z_std→1 are.
    Returns: dict[int, dict]
    """
    model.eval()
    per_class = {}
    with torch.no_grad():
        classes = torch.unique(c)
        for cls in classes:
            m = (c == cls)
            if not m.any():
                continue
            # Call your loss exactly as in training/validation, on the masked subset
            _, _, (rrmse_mean, rrmse_std) = flexible_flow_loss_from_model(
                model, x[m], c[m], **loss_kwargs
            )
            per_class[int(cls.item())] = {
                "rrmse_mean": round(float(rrmse_mean), 6),
                "rrmse_std":  round(float(rrmse_std), 6),
                "n": int(m.sum().item())
            }
    return per_class

def _format_iso_dict(iso_dict: dict[int, dict]) -> str:
    return {k: (v['rrmse_mean'], v['rrmse_std'], v['n']) for k, v in iso_dict.items()}.__str__()


def filter_flowpre_columns(df: pd.DataFrame, cols_to_exclude: list[str], condition_col: str) -> pd.DataFrame:
    """
    Removes specified columns from a DataFrame while preserving original column order,
    except for moving the condition column to the front. Raises error if condition column is missing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_exclude (list[str]): List of column names to remove.
        condition_col (str): Name of the condition column to move to front.

    Returns:
        pd.DataFrame: Filtered DataFrame with columns in original order (condition_col first).
    """
    if condition_col not in df.columns:
        raise ValueError(f"Condition column '{condition_col}' not found in DataFrame.")

    excluded_cols = set(cols_to_exclude or []).union({"post_cleaning_index"})

    if condition_col in excluded_cols:
        raise ValueError(f"Condition column '{condition_col}' is included in cols_to_exclude. This is not allowed.")

    # Preserve original order, move condition_col to the front
    feature_cols = [col for col in df.columns if col not in excluded_cols and col != condition_col]
    ordered_cols = [condition_col] + feature_cols

    return df[ordered_cols].copy()

def compute_latent_feature_influence(
    model, x, c, feature_names, influence_step_fraction=0.01, sweep_range=(-3, 3)
):
    """
    For each latent dimension z_i, estimate its influence on each raw feature x_j.

    Args:
        influence_step_fraction (float): Step size as a fraction of the total sweep range (e.g., 0.005 for 0.5%)

    Returns:
        dict: { "z_0": {"feature_1": (raw, norm), ...}, ... }
    """
    model.eval()
    device = model.device  # Assumes you store this in FlowPre

    with torch.no_grad():
        z, _ = model.forward(x.to(device), c.to(device))

    z = z.cpu()
    c = c.cpu()

    base_z = z.mean(dim=0)
    influence_dict = {}

    sweep_min, sweep_max = sweep_range
    total_range = abs(sweep_max - sweep_min)
    num_steps = int(round(1 / influence_step_fraction)) + 1
    sweep_vals = torch.linspace(sweep_min, sweep_max, steps=num_steps)

    for i in range(z.shape[1]):
        z_sweep = base_z.repeat(num_steps, 1)
        z_sweep[:, i] = sweep_vals

        c_repeat = c[:1].repeat(num_steps)

        x_rec = model.inverse(z_sweep.to(model.device), c_repeat.to(model.device))[0].detach().cpu().numpy()
        x_std = np.std(x_rec, axis=0)  # raw influence

        total = x_std.sum()
        x_norm = (x_std / total) if total > 0 else np.zeros_like(x_std)  # normalized

        # Ensure all features appear, even if 0
        influence_dict[f"z_{i}"] = {
            fname: (round(float(raw), 6), round(float(norm), 6))
            for fname, raw, norm in zip(feature_names, x_std, x_norm)
        }

    return influence_dict


def build_flow_pre_model(model_cfg: dict, input_dim: int, num_classes: int, device: str = "cpu") -> FlowPre:
    """
    Constructs a FlowPre model from config dict.

    Args:
        model_cfg (dict): Dictionary with model hyperparameters.
        input_dim (int): Number of input features.
        num_classes (int): Number of condition classes.
        device (str): "cpu" or "cuda".

    Returns:
        FlowPre instance.
    """
    return FlowPre(
        input_dim=input_dim,
        num_classes=num_classes,
        embedding_dim=model_cfg.get("embedding_dim", 8),
        hidden_features=model_cfg.get("hidden_features", 64),
        num_layers=model_cfg.get("num_layers", 2),
        use_actnorm=model_cfg.get("use_actnorm", True),
        use_learnable_permutations=model_cfg.get("use_learnable_permutations", True),
        num_bins=model_cfg.get("num_bins", 8),
        tail_bound=model_cfg.get("tail_bound", 3.0),
        initial_affine_layers=model_cfg.get("initial_affine_layers", 2),
        affine_rq_ratio=tuple(model_cfg.get("affine_rq_ratio", [1, 2])),
        n_repeat_blocks=model_cfg.get("n_repeat_blocks", 5),
        final_rq_layers=model_cfg.get("final_rq_layers", 3),
        lulinear_finisher=model_cfg.get("lulinear_finisher", True),
        device=device
    )

def prepare_flowpre_dataloader(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    condition_col: str,
    batch_size: int,
    device: torch.device,
    X_test: pd.DataFrame | None = None,
    seed: int | None = None,
):
    """
    Validates and prepares tensors and dataloader for FlowPre training.

    Returns:
        Tuple: (
            x_train: torch.Tensor,
            c_train: torch.Tensor,
            x_val: torch.Tensor,
            c_val: torch.Tensor,
            x_test: torch.Tensor | None,        # <--- NEW
            c_test: torch.Tensor | None,        # <--- NEW
            feature_names: List[str],
            train_dataset: TensorDataset,
            train_dataloader: DataLoader
        )
    """

    drop_cols = [condition_col]
    if "post_cleaning_index" in X_train.columns:
        drop_cols.append("post_cleaning_index")

    if condition_col not in X_train.columns or condition_col not in X_val.columns:
        raise ValueError(f"Missing '{condition_col}' column in X_train or X_val.")

    X_train_features = X_train.drop(columns=drop_cols).copy()
    X_val_features   = X_val.drop(columns=drop_cols).copy()

    if list(X_train_features.columns) != list(X_val_features.columns):
        raise ValueError("Feature columns mismatch between X_train and X_val — names or order are not identical.")

    feature_names = X_train_features.columns.tolist()

    x_train = torch.tensor(X_train_features.values, dtype=torch.float32).to(device)
    x_val   = torch.tensor(X_val_features.values,   dtype=torch.float32).to(device)

    c_train_raw = X_train[condition_col].values
    c_val_raw   = X_val[condition_col].values

    if c_train_raw.ndim != 1 or c_val_raw.ndim != 1:
        raise ValueError("Condition labels must be 1D arrays.")

    if len(c_train_raw) != len(x_train) or len(c_val_raw) != len(x_val):
        raise ValueError("Mismatch between condition labels and feature tensors.")

    c_train = torch.tensor(c_train_raw, dtype=torch.long).to(device)
    c_val   = torch.tensor(c_val_raw,   dtype=torch.long).to(device)

    if x_train.shape[1] != len(feature_names) or x_val.shape[1] != len(feature_names):
        raise ValueError("x_(train|val) does not match feature_names in dimensionality.")

    # --- Optional TEST tensors ---
    x_test, c_test = None, None
    if X_test is not None:
        if condition_col not in X_test.columns:
            raise ValueError(f"Missing '{condition_col}' column in X_test.")
        drop_cols_test = [condition_col]
        if "post_cleaning_index" in X_test.columns:
            drop_cols_test.append("post_cleaning_index")
        X_test_features = X_test.drop(columns=drop_cols_test).copy()

        # guard: feature schema must match train/val
        if list(X_test_features.columns) != feature_names:
            raise ValueError("Feature columns mismatch between X_train and X_test — names or order are not identical.")

        x_test = torch.tensor(X_test_features.values, dtype=torch.float32).to(device)
        c_test = torch.tensor(X_test[condition_col].values, dtype=torch.long).to(device)

    # --- DataLoader (deterministic if seeded earlier) ---
    train_dataset = TensorDataset(x_train, c_train)

    g = torch.Generator(device="cpu")
    gen_seed = int(seed if seed is not None else torch.initial_seed())
    g.manual_seed(gen_seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )

    return x_train, c_train, x_val, c_val, x_test, c_test, feature_names, train_dataset, train_dataloader


# --- Determinism helper (no-op if seed is None) ---
def _maybe_set_seed(seed: int | None) -> int:
    """
    If seed is None, sample a 64-bit seed and return it.
    If seed is provided, use it deterministically and return it.
    Torch allows up to 2**63-1; NumPy legacy seeding wants < 2**32.
    """

    # 1) Choose a seed
    if seed is None:
        # 64-bit random; keep within torch's recommended bound
        seed = secrets.randbits(64) % (2**63 - 1)

    # 2) Apply to Python/NumPy/Torch
    random.seed(seed)

    # NumPy legacy RNG expects 32-bit
    np_seed = int(seed % (2**32 - 1))  # keep in [0, 2**32-2]
    np.random.seed(np_seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return int(seed)


def train_flowpre_model(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        condition_col: str,
        X_test: pd.DataFrame | None = None,
        seed: int | None = None,
        config_filename: str = "flow_pre",
        base_name: str = "flow_pre_v1",
        device: str = "cuda",
        verbose: bool = True
):
    """
    Train a conditional normalizing flow on X_train with validation on X_val using FlowPre.

    Args:
        X_train (pd.DataFrame): Scaled training features with condition column.
        X_val (pd.DataFrame): Scaled validation features with condition column.
        condition_col (str): Name of the column with condition labels (categorical).
        config_filename (str): Path to YAML config file.

    Returns:
        FlowPre model (trained)
    """

    seed = _maybe_set_seed(seed)

    config = load_yaml_config(config_filename)
    model_cfg = config["model"]
    train_cfg = config["training"]
    interp_cfg = config.get("interpretability", {})

    device = torch.device("cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    # Use helper to validate inputs and prepare tensors + loader
    x_train, c_train, x_val, c_val, x_test, c_test, feature_names, dataset, dataloader = prepare_flowpre_dataloader(
        X_train=X_train,
        X_val=X_val,
        condition_col=condition_col,
        batch_size=train_cfg["batch_size"],
        device=device,
        X_test=X_test,
        seed=seed,
    )

    model = build_flow_pre_model(model_cfg, input_dim=x_train.shape[1], num_classes=c_train.max().item() + 1,
                                 device=device)

    if model_cfg.get("use_actnorm", False):
        model.eval()
        with torch.no_grad():
            try:
                first_batch_x, first_batch_c = next(iter(dataloader))
                _ = model(first_batch_x.to(device), first_batch_c.to(device))
            except StopIteration:
                raise ValueError("Empty dataloader – check your input data.")
        model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    loss_kwargs = {
        "use_nll": train_cfg.get("use_nll", True),
        "use_logdet_penalty": train_cfg.get("use_logdet_penalty", False),
        "logdet_penalty_weight": train_cfg.get("logdet_penalty_weight", 0.1),
        "use_logdet_sq": train_cfg.get("use_logdet_sq", True),
        "use_logdet_abs": train_cfg.get("use_logdet_abs", True),
        "ref_logdet": train_cfg.get("ref_logdet", 5.0),
        "logdet_scale_factor": train_cfg.get("logdet_scale_factor", True),
        "use_mean_penalty": train_cfg.get("use_mean_penalty", False),
        "mean_penalty_weight": train_cfg.get("mean_penalty_weight", 0.1),
        "use_mean_sq": train_cfg.get("use_mean_sq", True),
        "use_mean_abs": train_cfg.get("use_mean_abs", True),
        "use_std_penalty": train_cfg.get("use_std_penalty", False),
        "std_penalty_weight": train_cfg.get("std_penalty_weight", 0.1),
        "use_std_sq": train_cfg.get("use_std_sq", True),
        "use_std_abs": train_cfg.get("use_std_abs", True),
        "use_logpz_centering": train_cfg.get("use_logpz_centering", False),
        "logpz_centering_weight": train_cfg.get("logpz_centering_weight", 0.0),
        "logpz_target": train_cfg.get("logpz_target", None),
        "use_skew_penalty": train_cfg.get("use_skew_penalty", False),
        "skew_penalty_weight": train_cfg.get("skew_penalty_weight", 0.1),
        "use_skew_sq": train_cfg.get("use_skew_sq", True),
        "use_skew_abs": train_cfg.get("use_skew_abs", True),
        "use_kurtosis_penalty": train_cfg.get("use_kurtosis_penalty", False),
        "kurtosis_penalty_weight": train_cfg.get("kurtosis_penalty_weight", 0.1),
        "use_kurtosis_sq": train_cfg.get("use_kurtosis_sq", True),
        "use_kurtosis_abs": train_cfg.get("use_kurtosis_abs", True),
        "clamp_logabsdet_range": train_cfg.get("clamp_logabsdet_range", None),
    }

    # Training state
    best_val_loss = float("inf")
    best_model_state = None
    state_loss_epoch = None
    best_metrics = {
        "train_rrmse_mean": None,
        "train_rrmse_std": None,
        "val_rrmse_mean": None,
        "val_rrmse_std": None,
    }

    early_stopping_patience = train_cfg.get("early_stopping_patience", 40)
    lr_decay_patience = train_cfg.get("lr_decay_patience", 16)
    min_improvement = train_cfg.get("min_improvement", 0.04)
    min_improvement_floor = train_cfg.get("min_improvement_floor", 0.0025)
    lr_decay_factor = train_cfg.get("lr_decay_factor", 0.5)
    lr_patience_factor = train_cfg.get("lr_patience_factor", 0.8)

    epochs_no_improve = 0
    lr_decay_wait = 0
    lr = train_cfg["learning_rate"]
    lr_factor = 1
    patience_factor = 1
    initial_lr_decay_patience = lr_decay_patience

    total_epochs = train_cfg["num_epochs"]

    log_training = train_cfg.get("log_training", True)

    # Logging and directories
    versioned_dir, versioned_name, log_file_path, snapshots_dir = setup_training_logs_and_dirs(
        base_name, config_filename, config, verbose, train_cfg.get("save_states", False),
        train_cfg.get("log_training", True)
    )

    flowpre_log(f"Using device: {device}", log_training=log_training,
                filename_or_path=log_file_path, verbose=verbose)
    flowpre_log(f"🎲 Using seed: {seed}", log_training=log_training,
                filename_or_path=log_file_path, verbose=verbose)

    base = StandardNormal([x_train.shape[1]])
    z_test = torch.zeros(1, x_train.shape[1]).to(device)
    flowpre_log(f"✅ Sanity check — log_prob at zero: {base.log_prob(z_test).item()}",
                log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

    model.train()
    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        total_loss = 0
        epoch_rrmse_mean, epoch_rrmse_std = [], []
        diagnostics_accum = {}

        for batch_x, batch_c in dataloader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)

            loss, diagnostics, (mean_rrmse, std_rrmse) = flexible_flow_loss_from_model(
                model, batch_x, batch_c, **loss_kwargs
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            epoch_rrmse_mean.append(mean_rrmse)
            epoch_rrmse_std.append(std_rrmse)

            for k, v in diagnostics.items():
                diagnostics_accum.setdefault(k, []).append(v)

        avg_train_loss = total_loss / len(dataloader.dataset)
        avg_rrmse_mean = sum(epoch_rrmse_mean) / len(epoch_rrmse_mean)
        avg_rrmse_std = sum(epoch_rrmse_std) / len(epoch_rrmse_std)

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_loss, val_diagnostics, (val_rrmse_mean, val_rrmse_std) = flexible_flow_loss_from_model(
                model, x_val, c_val, **loss_kwargs
            )

        log_epoch_diagnostics(epoch, diagnostics_accum, log_file_path, verbose)

        flowpre_log(f"📉 Epoch {epoch + 1}/{total_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}",
                    filename_or_path=log_file_path, verbose=verbose)
        flowpre_log(f"🔍 RRMSE (Train): mean={avg_rrmse_mean:.4f}, std={avg_rrmse_std:.4f}",
                    filename_or_path=log_file_path, verbose=verbose)
        flowpre_log(f"🔍 RRMSE (Val):   mean={val_rrmse_mean:.4f}, std={val_rrmse_std:.4f}",
                    filename_or_path=log_file_path, verbose=verbose)

        improvement = (best_val_loss - val_loss) / (abs(best_val_loss) + 1e-8) if best_val_loss < float(
            "inf") else float("inf")
        flowpre_log(f"📈 Validation Improvement: {improvement:.4f}", filename_or_path=log_file_path, verbose=verbose)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_loss_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())
            best_metrics["train_rrmse_mean"] = avg_rrmse_mean
            best_metrics["train_rrmse_std"] = avg_rrmse_std
            best_metrics["val_rrmse_mean"] = val_rrmse_mean
            best_metrics["val_rrmse_std"] = val_rrmse_std

            if improvement >= min_improvement:
                epochs_no_improve = 0
                lr_decay_wait = 0

                if train_cfg.get("save_states", False):
                    snapshot_path = snapshots_dir / f"{versioned_name}_epoch{epoch + 1}_valloss{val_loss:.2f}.pt"
                    torch.save(best_model_state, snapshot_path)
                    flowpre_log(f"💾 Saved snapshot: {snapshot_path.name}", filename_or_path=log_file_path,
                                verbose=verbose)
        else:
            epochs_no_improve += 1
            lr_decay_wait += 1

            if lr_decay_wait >= lr_decay_patience:
                # LR decay
                last_lr = lr
                lr_factor *= lr_decay_factor
                lr = train_cfg["learning_rate"] * lr_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                flowpre_log(f"🔽 LR decay: {last_lr:.6f} → {lr:.6f}", filename_or_path=log_file_path, verbose=verbose)

                # Reduce patience and min improvement
                old_improvement = min_improvement
                min_improvement = max(min_improvement_floor, min_improvement * lr_decay_factor)
                if min_improvement < old_improvement:
                    flowpre_log(f"🔽 Reducing minimum improvement: {old_improvement:.4f} → {min_improvement:.4f}",
                                log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

                old_lr_decay_patience = lr_decay_patience
                patience_factor *= lr_patience_factor
                new_lr_decay_patience = max(5, int(math.ceil(initial_lr_decay_patience * patience_factor)))
                lr_decay_patience = new_lr_decay_patience
                if lr_decay_patience < old_lr_decay_patience:
                    flowpre_log(
                        f"🔽 Reducing learning rate decay patience: {old_lr_decay_patience} → {lr_decay_patience}",
                        log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

                if early_stopping_patience > 5:
                    epochs_no_improve = 0

                lr_decay_wait = 0

                if old_improvement * lr_decay_factor < min_improvement:
                    old_early_stopping_patience = early_stopping_patience
                    early_stopping_patience *= 0.5
                    early_stopping_patience = max(early_stopping_patience, 5)
                    if early_stopping_patience < old_early_stopping_patience:
                        flowpre_log(
                            f"🔽 Reducing early stopping patience: {old_early_stopping_patience} → {early_stopping_patience}",
                            log_training=log_training, filename_or_path=log_file_path, verbose=verbose)

            if epochs_no_improve >= early_stopping_patience:
                flowpre_log(f"🛌 Early stopping at epoch {epoch + 1}", filename_or_path=log_file_path, verbose=verbose)
                break

    flowpre_log(f"✅ Best validation loss: {best_val_loss:.4f} (epoch {state_loss_epoch})",
                filename_or_path=log_file_path, verbose=verbose)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final snapshot
    if train_cfg.get("save_states", False):
        final_path = snapshots_dir / f"{versioned_name}_best_epoch{state_loss_epoch}_valloss{best_val_loss:.2f}.pt"
        torch.save(best_model_state, final_path)

    # Final reconstruction metrics on both train and val sets
    model.eval()
    with torch.no_grad():
        # --- Train ---
        z_train, _ = model.forward(x_train, c_train)
        x_recon_train = model.inverse(z_train, c_train)[0]

        rrmse_train = torch.sqrt(torch.mean((x_recon_train - x_train) ** 2)) / torch.sqrt(torch.mean(x_train ** 2))
        rrmse_train = rrmse_train.item()

        ss_res_train = torch.sum((x_train - x_recon_train) ** 2)
        ss_tot_train = torch.sum((x_train - x_train.mean(dim=0)) ** 2)
        r2_train = (1 - ss_res_train / ss_tot_train).item()

        # --- Val ---
        z_val, _ = model.forward(x_val, c_val)
        x_recon_val = model.inverse(z_val, c_val)[0]

        rrmse_val = torch.sqrt(torch.mean((x_recon_val - x_val) ** 2)) / torch.sqrt(torch.mean(x_val ** 2))
        rrmse_val = rrmse_val.item()

        ss_res_val = torch.sum((x_val - x_recon_val) ** 2)
        ss_tot_val = torch.sum((x_val - x_val.mean(dim=0)) ** 2)
        r2_val = (1 - ss_res_val / ss_tot_val).item()

    flowpre_log(f"📈 Train RRMSE: {rrmse_train:.4f}, R²: {r2_train:.4f}", filename_or_path=log_file_path,
                verbose=verbose)
    flowpre_log(f"📈 Val   RRMSE: {rrmse_val:.4f}, R²: {r2_val:.4f}", filename_or_path=log_file_path, verbose=verbose)

    influence_dict = compute_latent_feature_influence(
        model, x_train, c_train, feature_names,
        influence_step_fraction=interp_cfg.get("influence_step_fraction", 0.01),
        sweep_range=tuple(interp_cfg.get("sweep_range", [-model_cfg["tail_bound"], model_cfg["tail_bound"]]))
    )

    # --- Per-class isotropy RRMSE from the loss (using full Train/Val tensors) ---
    train_iso_per_class = _iso_rrmse_per_class_from_loss(model, x_train, c_train, loss_kwargs)
    val_iso_per_class = _iso_rrmse_per_class_from_loss(model, x_val, c_val, loss_kwargs)
    test_iso_per_class = _iso_rrmse_per_class_from_loss(model, x_test, c_test,
                                                        loss_kwargs) if x_test is not None else {}

    # --- Whole-set isotropy diagnostics (moments, Mahalanobis, eigvals) ---
    train_iso_whole = _latent_isotropy_stats(model, x_train, c_train)
    val_iso_whole = _latent_isotropy_stats(model, x_val, c_val)
    test_iso_whole = _latent_isotropy_stats(model, x_test, c_test) if x_test is not None else {}

    # --- Per-class extended isotropy diagnostics ---
    train_iso_per_class_ext = _latent_isotropy_stats_per_class(model, x_train, c_train)
    val_iso_per_class_ext = _latent_isotropy_stats_per_class(model, x_val, c_val)
    test_iso_per_class_ext = _latent_isotropy_stats_per_class(model, x_test, c_test) if x_test is not None else {}

    # Optional compact log
    flowpre_log(
        f"📊 Per-class isotropy (Train): {_format_iso_dict(train_iso_per_class)}",
        filename_or_path=log_file_path, verbose=verbose
    )
    flowpre_log(
        f"📊 Per-class isotropy (Val):   {_format_iso_dict(val_iso_per_class)}",
        filename_or_path=log_file_path, verbose=verbose
    )

    # --- Overall isotropy on TRAIN/VAL/TEST from the loss (mean→0, std→1) ---
    train_rrmse_mean_whole, train_rrmse_std_whole = None, None
    val_rrmse_mean_whole, val_rrmse_std_whole = None, None
    test_rrmse_mean_whole, test_rrmse_std_whole = None, None

    model.eval()
    with torch.no_grad():
        # overall isotropy on TRAIN
        _, _, (train_rrmse_mean_whole, train_rrmse_std_whole) = flexible_flow_loss_from_model(
            model, x_train, c_train, **loss_kwargs
        )
        # overall isotropy on VAL
        _, _, (val_rrmse_mean_whole, val_rrmse_std_whole) = flexible_flow_loss_from_model(
            model, x_val, c_val, **loss_kwargs
        )
        # overall isotropy on TEST (if provided)
        if x_test is not None:
            _, _, (test_rrmse_mean_whole, test_rrmse_std_whole) = flexible_flow_loss_from_model(
                model, x_test, c_test, **loss_kwargs
            )

    # cast to plain floats
    train_rrmse_mean_whole = float(train_rrmse_mean_whole)
    train_rrmse_std_whole = float(train_rrmse_std_whole)
    val_rrmse_mean_whole = float(val_rrmse_mean_whole)
    val_rrmse_std_whole = float(val_rrmse_std_whole)
    if x_test is not None:
        test_rrmse_mean_whole = float(test_rrmse_mean_whole)
        test_rrmse_std_whole = float(test_rrmse_std_whole)

    # --- Save results with overall isotropy (not per-epoch best_metrics) ---
    if train_cfg.get("save_results", False):
        results = {
            "best_epoch": state_loss_epoch,
            "total_epochs": epoch + 1,
            "seed": seed,
            "train": {
                "rrmse_mean_whole": round(train_rrmse_mean_whole, 6),
                "rrmse_std_whole": round(train_rrmse_std_whole, 6),

                "per_class_iso_rrmse": train_iso_per_class,
                "isotropy_stats": train_iso_whole,
                "isotropy_stats_per_class": train_iso_per_class_ext,
            },
            "val": {
                "rrmse_mean_whole": round(val_rrmse_mean_whole, 6),
                "rrmse_std_whole": round(val_rrmse_std_whole, 6),

                "per_class_iso_rrmse": val_iso_per_class,
                "isotropy_stats": val_iso_whole,
                "isotropy_stats_per_class": val_iso_per_class_ext,
            }
        }

        if x_test is not None:
            results["test"] = {
                "rrmse_mean_whole": round(test_rrmse_mean_whole, 6) if test_rrmse_mean_whole is not None else None,
                "rrmse_std_whole": round(test_rrmse_std_whole, 6) if test_rrmse_std_whole is not None else None,
                "per_class_iso_rrmse": test_iso_per_class,
                "isotropy_stats": test_iso_whole,
                "isotropy_stats_per_class": test_iso_per_class_ext,
            }

        results_path = versioned_dir / f"{versioned_name}_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, sort_keys=False)
        flowpre_log(f"📝 Saved enriched results to: {results_path}", filename_or_path=log_file_path, verbose=verbose)

    # Always save the config
    shutil.copy(ROOT_PATH / "config" / config_filename, versioned_dir / f"{versioned_name}.yaml")

    # Save model only if enabled
    if train_cfg.get("save_model", False):
        torch.save(model.state_dict(), versioned_dir / f"{versioned_name}.pt")
        flowpre_log(f"💾 Model saved: {versioned_name}.pt", filename_or_path=log_file_path, verbose=verbose)

    # Log influence only if saved
    if interp_cfg.get("save_influence", False):
        with open(versioned_dir / f"{versioned_name}_influence.json", "w") as f:
            json.dump(influence_dict, f, indent=2)
        flowpre_log(f"🧠 Influence saved: {versioned_name}_influence.json", filename_or_path=log_file_path,
                    verbose=verbose)

    flowpre_log(f"✅ Config saved under {versioned_dir}", filename_or_path=log_file_path, verbose=verbose)

    return model


def train_flowpre_pipeline(
    condition_col: str = "type",
    cols_to_exclude: list[str] = None,
    config_filename: str = "flow_pre.yaml",
    base_name: str = "flow_pre",
    device: str = "cuda",
    seed: int | None = None,
    verbose: bool = True
):
    """
    Complete pipeline to load, filter and train a FlowPre model using default splits.

    Args:
        condition_col (str): Name of the condition column (e.g., 'type').
        cols_to_exclude (list[str]): Columns to exclude from training (e.g., ['post_cleaning_index']).
        config_filename (str): YAML config file for model/training.
        base_name (str): Base name for logging/saving the model.
        verbose (bool): Whether to log progress.

    Returns:
        Trained FlowPre model.
    """
    if cols_to_exclude is None:
        cols_to_exclude = ["post_cleaning_index"]

    # Step 1: Load train/val/test splits   <-- CHANGE (was only train/val)
    X_train, X_val, X_test, *_ = load_or_create_raw_splits(
        condition_col=condition_col,
        verbose=verbose
    )

    # Step 2: Filter and align columns
    X_train_filtered = filter_flowpre_columns(X_train, cols_to_exclude, condition_col)
    X_val_filtered = filter_flowpre_columns(X_val, cols_to_exclude, condition_col)
    X_test_filtered = filter_flowpre_columns(X_test, cols_to_exclude, condition_col)

    # Step 3: Train the model
    model = train_flowpre_model(
        X_train=X_train_filtered,
        X_val=X_val_filtered,
        X_test=X_test_filtered,
        condition_col=condition_col,
        config_filename=config_filename,
        base_name=base_name,
        device=device,
        seed=seed,
        verbose=verbose
    )

    return model

def encode_with_flowpre_model(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    condition_col: str,
    cols_to_exclude: list[str]
) -> pd.DataFrame:
    """
    Applies FlowPre model forward pass to transform df into latent space.

    Returns a DataFrame with z_i columns + condition_col (and post_cleaning_index if present).
    """
    df_filtered = filter_flowpre_columns(df, cols_to_exclude, condition_col)

    x = torch.tensor(df_filtered.drop(columns=[condition_col]).values, dtype=torch.float32).to(device)
    c = torch.tensor(df_filtered[condition_col].values, dtype=torch.long).to(device)

    with torch.no_grad():
        z, _ = model.forward(x, c)

    z_names = [f"z_{i}" for i in range(z.shape[1])]
    df_latent = pd.DataFrame(z.cpu().numpy(), columns=z_names, index=df_filtered.index)
    df_latent[condition_col] = df_filtered[condition_col].values

    ordered_cols = [condition_col] + z_names
    if "post_cleaning_index" in df.columns:
        df_latent.insert(0, "post_cleaning_index", df.loc[df_latent.index, "post_cleaning_index"])
        ordered_cols = ["post_cleaning_index"] + ordered_cols

    return df_latent[ordered_cols]

def transform_to_latent_with_flowpre(
    condition_col: str,
    cols_to_exclude: list[str],
    model_name: str = "flow_pre_v1",
    X_train: Optional[pd.DataFrame] = None,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    device: str = "cuda",
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforms input datasets into latent Gaussian space using an existing FlowPre model.

    Args:
        condition_col (str): Column used for conditioning.
        cols_to_exclude (list[str]): Columns to drop before encoding.
        model_name (str): Name of trained FlowPre model (must end with _vX).
        X_train, X_val, X_test (pd.DataFrame or None): Datasets to transform.
        verbose (bool): Whether to log steps.

    Returns:
        Tuple of latent DataFrames: (X_train_latent, X_val_latent, X_test_latent or None)
    """
    assert re.match(r".+_v\d+$", model_name), f"❌ model_name '{model_name}' must end with '_vX'"

    model_dir = ROOT_PATH / "outputs" / "models" / "flow_pre" / model_name
    model_path = model_dir / f"{model_name}.pt"
    config_path = model_dir / f"{model_name}.yaml"

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"❌ Model or config not found for: {model_name}")

    flowpre_log(f"📦 Loading FlowPre model from: {model_path}", verbose=verbose)
    flowpre_log(f"⚙️  Loading config from: {config_path}", verbose=verbose)

    config = load_yaml_config(config_path)
    model_cfg = config["model"]
    device = torch.device("cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    # Load splits if not provided
    if X_train is None or X_val is None or X_test is None:
        X_train, X_val, X_test, *_ = load_or_create_raw_splits(condition_col=condition_col, verbose=verbose)

    # Use X_train to determine input dimensions
    df_tmp_filtered = filter_flowpre_columns(X_train, cols_to_exclude, condition_col)
    input_dim = df_tmp_filtered.drop(columns=[condition_col]).shape[1]
    num_classes = df_tmp_filtered[condition_col].nunique()

    model = build_flow_pre_model(model_cfg, input_dim=input_dim, num_classes=num_classes, device=device)
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    X_train_latent = encode_with_flowpre_model(X_train, model, device, condition_col, cols_to_exclude)
    X_val_latent = encode_with_flowpre_model(X_val, model, device, condition_col, cols_to_exclude)
    X_test_latent = encode_with_flowpre_model(X_test, model, device, condition_col, cols_to_exclude)

    for name, original_df, latent_df in [
        ("X_train", X_train, X_train_latent),
        ("X_val", X_val, X_val_latent),
        ("X_test", X_test, X_test_latent),
    ]:
        if original_df is not None:
            # Calculate latent dimensionality (input_dim - excluded - condition)
            expected_latent_dim = (
                original_df.drop(columns=cols_to_exclude)
                .drop(columns=[condition_col])
                .shape[1]
            )
            expected_cols = 1 + len(cols_to_exclude) + expected_latent_dim

            if original_df.shape[0] != latent_df.shape[0] or latent_df.shape[1] != expected_cols:
                raise ValueError(
                    f"❌ {name} shape mismatch: "
                    f"expected ({original_df.shape[0]}, {expected_cols}), "
                    f"got {latent_df.shape}"
                )

            # Validate first N columns match: excluded + condition_col
            expected_prefix = cols_to_exclude + [condition_col]
            actual_prefix = list(latent_df.columns[: len(expected_prefix)])
            if actual_prefix != expected_prefix:
                raise ValueError(
                    f"❌ {name} column prefix mismatch: expected {expected_prefix}, got {actual_prefix}"
                )
    flowpre_log(f"Job finished! 👍", verbose=verbose)
    return X_train_latent, X_val_latent, X_test_latent



