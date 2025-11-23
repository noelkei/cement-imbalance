from data.utils import log, load_column_mapping_by_group, load_type_mapping
from training.utils import ROOT_PATH
from data.splits import prepare_splits
from sklearn.utils import shuffle
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler

from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple, Optional

_SCALED_ROOT = Path(ROOT_PATH) / "data" / "sets" / "scaled_sets"
_MODELS_DIR  = Path(ROOT_PATH) / "outputs" / "models" / "flow_pre"

def load_or_create_raw_splits(
    df_name: str = "df_input",
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    verbose: bool = True,
    force: bool = False
):
    """
    Load or create X/y/removed sets for train, val, test based on 'post_cleaning_index', stratified by condition_col.

    Args:
        df_name (str): Folder name under data/sets/
        condition_col (str): Column used for stratified splitting
        val_size (int): Max val samples per class
        test_size (int): Max test samples per class
        target (str): Target passed to prepare_splits
        verbose (bool): Log steps
        force (bool): Recreate sets even if they exist

    Returns:
        9 DataFrames: X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test
    """

    sets_base = ROOT_PATH / "data" / "sets" / df_name
    sets_X_dir = sets_base / "X"
    sets_y_dir = sets_base / "y"
    sets_r_dir = sets_base / "removed"

    sets_X_dir.mkdir(parents=True, exist_ok=True)
    sets_y_dir.mkdir(parents=True, exist_ok=True)
    sets_r_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train": {
            "X": f"{df_name}_X_train.csv",
            "y": f"{df_name}_y_train.csv",
            "r": f"{df_name}_r_train.csv"
        },
        "val": {
            "X": f"{df_name}_X_val.csv",
            "y": f"{df_name}_y_val.csv",
            "r": f"{df_name}_r_val.csv"
        },
        "test": {
            "X": f"{df_name}_X_test.csv",
            "y": f"{df_name}_y_test.csv",
            "r": f"{df_name}_r_test.csv"
        },
    }

    filepaths = {
        split: {
            "X": sets_X_dir / filenames[split]["X"],
            "y": sets_y_dir / filenames[split]["y"],
            "r": sets_r_dir / filenames[split]["r"]
        } for split in ["train", "val", "test"]
    }

    if all(p["X"].exists() and p["y"].exists() and p["r"].exists() for p in filepaths.values()) and not force:
        log(f"📁 All sets for '{df_name}' already exist. Loading from disk...", verbose)
        X_train = pd.read_csv(filepaths["train"]["X"])
        X_val   = pd.read_csv(filepaths["val"]["X"])
        X_test  = pd.read_csv(filepaths["test"]["X"])
        y_train = pd.read_csv(filepaths["train"]["y"])
        y_val   = pd.read_csv(filepaths["val"]["y"])
        y_test  = pd.read_csv(filepaths["test"]["y"])
        r_train = pd.read_csv(filepaths["train"]["r"])
        r_val   = pd.read_csv(filepaths["val"]["r"])
        r_test  = pd.read_csv(filepaths["test"]["r"])
        return X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test

    log(f"🔄 Creating sets for {df_name} (val={val_size}, test={test_size})...", verbose)

    df_X, df_y, df_r = prepare_splits(target=target, verbose=verbose, force=force)

    df_X = df_X.sort_values("post_cleaning_index").reset_index(drop=True)
    df_y = df_y.sort_values("post_cleaning_index").reset_index(drop=True)
    df_r = df_r.sort_values("post_cleaning_index").reset_index(drop=True)

    used_indices = set()
    val_indices = []
    test_indices = []

    for cond in df_X[condition_col].unique():
        cond_indices = df_X[df_X[condition_col] == cond]["post_cleaning_index"].tolist()
        cond_indices = [i for i in cond_indices if i not in used_indices]

        n_val = min(val_size, len(cond_indices))
        n_test = min(test_size, len(cond_indices) - n_val)

        selected = shuffle(cond_indices, random_state=42)
        val_indices.extend(selected[:n_val])
        test_indices.extend(selected[n_val:n_val + n_test])
        used_indices.update(selected[:n_val + n_test])

    val_mask = df_X["post_cleaning_index"].isin(val_indices)
    test_mask = df_X["post_cleaning_index"].isin(test_indices)
    train_mask = ~(val_mask | test_mask)

    sets_X = {
        "train": df_X[train_mask],
        "val": df_X[val_mask],
        "test": df_X[test_mask]
    }

    sets_y = {
        "train": df_y[df_y["post_cleaning_index"].isin(sets_X["train"]["post_cleaning_index"])],
        "val": df_y[df_y["post_cleaning_index"].isin(sets_X["val"]["post_cleaning_index"])],
        "test": df_y[df_y["post_cleaning_index"].isin(sets_X["test"]["post_cleaning_index"])]
    }

    sets_r = {
        "train": df_r[df_r["post_cleaning_index"].isin(sets_X["train"]["post_cleaning_index"])],
        "val": df_r[df_r["post_cleaning_index"].isin(sets_X["val"]["post_cleaning_index"])],
        "test": df_r[df_r["post_cleaning_index"].isin(sets_X["test"]["post_cleaning_index"])]
    }

    # Sort and save
    for split in ["train", "val", "test"]:
        for d in [sets_X, sets_y, sets_r]:
            d[split] = d[split].sort_values("post_cleaning_index").reset_index(drop=True)

        sets_X[split].to_csv(filepaths[split]["X"], index=False)
        sets_y[split].to_csv(filepaths[split]["y"], index=False)
        sets_r[split].to_csv(filepaths[split]["r"], index=False)

        log(f"✅ Saved {split} set → X: {sets_X[split].shape}, y: {sets_y[split].shape}, removed: {sets_r[split].shape}", verbose)

    return (
        sets_X["train"], sets_X["val"], sets_X["test"],
        sets_y["train"], sets_y["val"], sets_y["test"],
        sets_r["train"], sets_r["val"], sets_r["test"]
    )

def plot_umap_and_histograms_per_set(X_train, X_val, X_test, y_train, y_val, y_test, n_neighbors=15, min_dist=0.1, random_state=42):
    def clean_input(X):
        return X.drop(columns=["post_cleaning_index", "type"], errors="ignore")

    # Step 1: Fit UMAP on all combined sets (same projection space)
    X_all = pd.concat([clean_input(X_train), clean_input(X_val), clean_input(X_test)], axis=0)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reducer.fit(X_all)

    # Step 2: Transform each set
    sets = {
        "Train": (X_train, reducer.transform(clean_input(X_train))),
        "Validation": (X_val, reducer.transform(clean_input(X_val))),
        "Test": (X_test, reducer.transform(clean_input(X_test)))
    }

    # Step 3: Plot UMAP per set
    for name, (X_orig, embedding) in sets.items():
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=X_orig["type"], cmap="tab10", s=20, alpha=0.7
        )
        plt.colorbar(scatter, label="Type")
        plt.title(f"UMAP - {name} Set")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Step 4: Plot histograms of 'init' target column
    plt.figure(figsize=(10, 4))
    sns.histplot(y_train["init"], kde=False, bins=50, label="Train", color="blue", stat="density", alpha=0.5)
    sns.histplot(y_val["init"], kde=False, bins=50, label="Val", color="green", stat="density", alpha=0.5)
    sns.histplot(y_test["init"], kde=False, bins=50, label="Test", color="red", stat="density", alpha=0.5)
    plt.title("Histogram of Target Column: 'init'")
    plt.legend()
    plt.grid(True)
    plt.xticks([])
    plt.tight_layout()
    plt.show()

def show_type_distribution(X_train, X_val, X_test):
    print("🔹 Number of samples per 'type' in each set:\n")

    print("📦 Train set:")
    print(X_train["type"].value_counts().sort_index(), "\n")

    print("📦 Validation set:")
    print(X_val["type"].value_counts().sort_index(), "\n")

    print("📦 Test set:")
    print(X_test["type"].value_counts().sort_index(), "\n")

    # Optionally return the counts if you want to reuse them
    return (
        X_train["type"].value_counts(),
        X_val["type"].value_counts(),
        X_test["type"].value_counts()
    )

def check_index_alignment(X_train, y_train, removed_train,
                          X_val, y_val, removed_val,
                          X_test, y_test, removed_test):
    def check_set(name, x_df, y_df, r_df):
        x_idx = x_df["post_cleaning_index"]
        y_idx = y_df["post_cleaning_index"]
        r_idx = r_df["post_cleaning_index"]

        x_match_y = x_idx.equals(y_idx)
        x_match_r = x_idx.equals(r_idx)

        print(f"\n🔎 Checking set: {name}")
        print(f"   📐 X vs y index match: {'✅' if x_match_y else '❌'}")
        print(f"   📐 X vs removed index match: {'✅' if x_match_r else '❌'}")

        if not x_match_y:
            mismatches = x_idx.compare(y_idx)
            print("   ❌ X vs y mismatched indices:\n", mismatches.head())
        if not x_match_r:
            mismatches = x_idx.compare(r_idx)
            print("   ❌ X vs removed mismatched indices:\n", mismatches.head())

    check_set("Train", X_train, y_train, removed_train)
    check_set("Validation", X_val, y_val, removed_val)
    check_set("Test", X_test, y_test, removed_test)

def compute_rrmse_mean_std(df1: pd.DataFrame, df2: pd.DataFrame, exclude_cols: list = ["post_cleaning_index", "type", "datum", "date", "process"], scaled: bool = False) -> dict:
    """
    Compute the RRMSE between the means and standard deviations of matching columns from two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        exclude_cols (list): Columns to exclude from comparison.

    Returns:
        dict: {
            "mean_rrmse": float,
            "std_rrmse": float
        }
    """
    common_cols = [col for col in df1.columns if col in df2.columns and col not in exclude_cols]

    df1_means = df1[common_cols].mean()
    df2_means = df2[common_cols].mean()
    df1_stds = df1[common_cols].std()
    df2_stds = df2[common_cols].std()

    # Mean RRMSE
    mean_rmse = np.sqrt(np.mean((df1_means - df2_means) ** 2))
    mean_denom = np.mean(np.abs(df1_means)) if not scaled else 1
    mean_rrmse = mean_rmse / mean_denom if mean_denom != 0 else np.nan

    # Std RRMSE
    std_rmse = np.sqrt(np.mean((df1_stds - df2_stds) ** 2))
    std_denom = np.mean(np.abs(df1_stds))
    std_rrmse = std_rmse / std_denom if std_denom != 0 else np.nan

    return {
        "mean_rrmse": mean_rrmse,
        "std_rrmse": std_rrmse
    }

def plot_scaled_histograms(X, bins=100, exclude_cols=["post_cleaning_index", "type"]):
    features = [col for col in X.columns if col not in exclude_cols]

    # Fit scalers on the entire dataset
    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()

    X_standard = pd.DataFrame(scaler_standard.fit_transform(X[features]), columns=features)
    X_robust = pd.DataFrame(scaler_robust.fit_transform(X[features]), columns=features)

    for feature in features:
        plt.figure(figsize=(15, 4))

        # Original
        plt.subplot(1, 3, 1)
        plt.hist(X[feature], bins=bins, color="skyblue", edgecolor="black", alpha=0.7, density=True)
        plt.title(f"Original: {feature}")
        plt.grid(True)
        plt.xticks([])

        # Standard scaled
        plt.subplot(1, 3, 2)
        plt.hist(X_standard[feature], bins=bins, color="orange", edgecolor="black", alpha=0.7, density=True)
        plt.title("Standard Scaler")
        plt.grid(True)

        # Robust scaled
        plt.subplot(1, 3, 3)
        plt.hist(X_robust[feature], bins=bins, color="green", edgecolor="black", alpha=0.7, density=True)
        plt.title("Robust Scaler")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def get_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    elif name == "robust":
        return RobustScaler()
    elif name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    elif name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"❌ Unsupported scaler: {name}. Choose from 'standard', 'robust', 'quantile', 'minmax'.")

def load_or_create_scaled_sets(
    raw_df_name: str = "df_input",
    scaled_df_name: str = "df_scaled",
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    force: bool = False,
    verbose: bool = True,
    x_scaler_type: str = "standard",
    y_scaler_type: str = "standard",
    exclude_cols: list = None
):
    if exclude_cols is None:
        exclude_cols = ["post_cleaning_index", "type"]

    x_scaler_type = x_scaler_type.lower()
    y_scaler_type = y_scaler_type.lower()

    allowed_x_scalers = ["standard", "robust", "quantile"]
    allowed_y_scalers = allowed_x_scalers + ["minmax"]

    if x_scaler_type not in allowed_x_scalers:
        raise ValueError(f"❌ Invalid X scaler type: {x_scaler_type}. Allowed: {allowed_x_scalers}")
    if y_scaler_type not in allowed_y_scalers:
        raise ValueError(f"❌ Invalid Y scaler type: {y_scaler_type}. Allowed: {allowed_y_scalers}")

    scaled_suffix = f"{scaled_df_name}_x{x_scaler_type}_y{y_scaler_type}"
    sets_base = ROOT_PATH / "data" / "sets" / "scaled_sets" / scaled_suffix

    dirs = {
        "X": sets_base / "X",
        "y": sets_base / "y",
        "r": sets_base / "removed",
        "scalers": sets_base / "scalers"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    filenames = {
        split: {
            "X": dirs["X"] / f"{scaled_suffix}_X_{split}.csv",
            "y": dirs["y"] / f"{scaled_suffix}_y_{split}.csv",
            "r": dirs["r"] / f"{scaled_suffix}_r_{split}.csv"
        } for split in ["train", "val", "test"]
    }

    scaler_paths = {
        "X": dirs["scalers"] / f"{scaled_suffix}_X_scaler.pkl",
        "y": dirs["scalers"] / f"{scaled_suffix}_y_scaler.pkl"
    }

    if all(
        filenames[split]["X"].exists() and
        filenames[split]["y"].exists() and
        filenames[split]["r"].exists()
        for split in ["train", "val", "test"]
    ) and all(p.exists() for p in scaler_paths.values()) and not force:
        log(f"📦 Scaled sets already exist for '{scaled_suffix}'. Loading from disk...", verbose)

        return (
            pd.read_csv(filenames["train"]["X"]),
            pd.read_csv(filenames["val"]["X"]),
            pd.read_csv(filenames["test"]["X"]),
            pd.read_csv(filenames["train"]["y"]),
            pd.read_csv(filenames["val"]["y"]),
            pd.read_csv(filenames["test"]["y"]),
            pd.read_csv(filenames["train"]["r"]),
            pd.read_csv(filenames["val"]["r"]),
            pd.read_csv(filenames["test"]["r"]),
            joblib.load(scaler_paths["X"]),
            joblib.load(scaler_paths["y"])
        )

    # Load raw sets
    X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test = load_or_create_raw_splits(
        df_name=raw_df_name,
        condition_col=condition_col,
        val_size=val_size,
        test_size=test_size,
        target=target,
        verbose=verbose,
        force=force
    )

    X_cols = [col for col in X_train.columns if col not in exclude_cols]
    y_cols = [col for col in y_train.columns if col not in exclude_cols]

    # Fit scalers
    x_scaler = get_scaler(x_scaler_type).fit(X_train[X_cols])
    y_scaler = get_scaler(y_scaler_type).fit(y_train[y_cols])

    for split, X_df, y_df, r_df in zip(
        ["train", "val", "test"],
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        [r_train, r_val, r_test]
    ):
        X_scaled = X_df.copy()
        y_scaled = y_df.copy()

        X_scaled[X_cols] = x_scaler.transform(X_df[X_cols])
        y_scaled[y_cols] = y_scaler.transform(y_df[y_cols])

        X_scaled.to_csv(filenames[split]["X"], index=False)
        y_scaled.to_csv(filenames[split]["y"], index=False)
        r_df.to_csv(filenames[split]["r"], index=False)

        log(f"✅ Saved {split} → X: {X_scaled.shape}, y: {y_scaled.shape}, r: {r_df.shape}", verbose)

    joblib.dump(x_scaler, scaler_paths["X"])
    joblib.dump(y_scaler, scaler_paths["y"])
    log(f"💾 Scalers saved to {scaler_paths['X']} and {scaler_paths['y']}", verbose)

    return (
        pd.read_csv(filenames["train"]["X"]),
        pd.read_csv(filenames["val"]["X"]),
        pd.read_csv(filenames["test"]["X"]),
        pd.read_csv(filenames["train"]["y"]),
        pd.read_csv(filenames["val"]["y"]),
        pd.read_csv(filenames["test"]["y"]),
        pd.read_csv(filenames["train"]["r"]),
        pd.read_csv(filenames["val"]["r"]),
        pd.read_csv(filenames["test"]["r"]),
        x_scaler,
        y_scaler
    )

def inverse_transform_and_compare(original_df, scaled_df, scaler, exclude_cols, name=""):
    original = original_df.copy()
    scaled = scaled_df.copy()

    cols_to_scale = [col for col in original.columns if col not in exclude_cols]
    original = original[cols_to_scale].reset_index(drop=True)
    recovered = pd.DataFrame(
        scaler.inverse_transform(scaled[cols_to_scale]),
        columns=cols_to_scale
    )

    recovered = recovered[original.columns]
    metrics = compute_rrmse_mean_std(original, recovered)
    print(f"🔁 Recovery check: {name}")
    print(f"   RRMSE of Mean: {metrics['mean_rrmse']:.6f}")
    print(f"   RRMSE of Std:  {metrics['std_rrmse']:.6f}")
    print()

def _flowpre_suffix(meta_tag: str, y_scaler_name: str) -> str:
    # e.g. "df_scaled_x_flowpre_rrmse_yminmax"
    return f"df_scaled_x_flowpre_{meta_tag}_y{y_scaler_name}"

def _flowpre_dirs(meta_tag: str, y_scaler_name: str) -> Dict[str, Path]:
    base = _SCALED_ROOT / _flowpre_suffix(meta_tag, y_scaler_name)
    return {
        "base":    base,
        "X":       base / "X",
        "y":       base / "y",
        "removed": base / "removed",
        "scalers": base / "scalers",
        "meta":    base / "meta",
        "model":   base / "model",     # ← store copied YAML + .pt here
    }

def _ensure_dirs(dirs: Dict[str, Path]) -> None:
    for k, p in dirs.items():
        if k == "base":
            continue
        p.mkdir(parents=True, exist_ok=True)

def _locate_model_artifacts(model_name: str) -> Tuple[Path, Path]:
    """
    Return (config_yaml, weights_pt) from outputs/models/flow_pre/<model_name>/.
    Tries exact filenames first, then falls back to first match.
    """
    model_dir = _MODELS_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    cfg = model_dir / f"{model_name}.yaml"
    if not cfg.exists():
        # fallback: first .yaml that doesn't look like results/influence
        cands = [p for p in model_dir.glob("*.yaml") if "_results" not in p.stem]
        if not cands:
            raise FileNotFoundError(f"No YAML config found under {model_dir}")
        cfg = sorted(cands)[0]

    pt = model_dir / f"{model_name}.pt"
    if not pt.exists():
        pts = list(model_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt weights found under {model_dir}")
        pt = sorted(pts)[0]

    return cfg, pt

def _copy_model_artifacts(model_name: str, dest_dir: Path) -> Dict[str, str]:
    """
    Copy YAML + .pt into dest_dir. Returns dict with their relative paths.
    If files already exist with same name, they are left as-is.
    """
    cfg_src, pt_src = _locate_model_artifacts(model_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cfg_dst = dest_dir / cfg_src.name
    pt_dst  = dest_dir / pt_src.name

    if not cfg_dst.exists():
        shutil.copy2(cfg_src, cfg_dst)
    if not pt_dst.exists():
        shutil.copy2(pt_src, pt_dst)

    return {
        "config": str(cfg_dst),
        "weights": str(pt_dst),
        "source_dir": str((_MODELS_DIR / model_name).resolve()),
    }

def _set_has_all_core_files(dirs: Dict[str, Path], suffix: str) -> bool:
    """Check X/Y/removed CSVs, scaler, and presence of model artifacts (yaml + .pt)."""
    required_csvs = [
        dirs["X"] / f"{suffix}_X_train.csv",
        dirs["X"] / f"{suffix}_X_val.csv",
        dirs["X"] / f"{suffix}_X_test.csv",
        dirs["y"] / f"{suffix}_y_train.csv",
        dirs["y"] / f"{suffix}_y_val.csv",
        dirs["y"] / f"{suffix}_y_test.csv",
        dirs["removed"] / f"{suffix}_r_train.csv",
        dirs["removed"] / f"{suffix}_r_val.csv",
        dirs["removed"] / f"{suffix}_r_test.csv",
        dirs["scalers"] / f"{suffix}_y_scaler.pkl",
    ]
    csv_ok = all(p.exists() for p in required_csvs)
    # model dir must contain at least one yaml and one pt
    has_yaml = any(dirs["model"].glob("*.yaml"))
    has_pt   = any(dirs["model"].glob("*.pt"))
    return csv_ok and has_yaml and has_pt

def transform_X_with_flowpre_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    condition_col: str = "type",
    cols_to_exclude: Optional[List[str]] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uses training.transform_to_latent_with_flowpre to encode X into latent z.
    Returns latent DataFrames for train/val/test (with condition_col and optional post_cleaning_index preserved).
    """
    if cols_to_exclude is None:
        cols_to_exclude = ["post_cleaning_index"]

    # ⬇️ Lazy import to break the cycle
    from training.train_flow_pre import transform_to_latent_with_flowpre

    X_lat_train, X_lat_val, X_lat_test = transform_to_latent_with_flowpre(
        condition_col=condition_col,
        cols_to_exclude=cols_to_exclude,
        model_name=model_name,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        device=device,
        verbose=verbose
    )
    return X_lat_train, X_lat_val, X_lat_test

def save_flowpre_scaled_bundle(
    X_lat_train: pd.DataFrame, X_lat_val: pd.DataFrame, X_lat_test: pd.DataFrame,
    y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame,
    r_train: pd.DataFrame, r_val: pd.DataFrame, r_test: pd.DataFrame,
    y_scaler_name: str,
    meta_tag: str,           # "rrmse" | "mvn" | "fair"
    model_name: str,         # subfolder in outputs/models/flow_pre
    condition_col: str = "type",
    verbose: bool = True,
) -> Tuple[Path, dict]:
    """
    Saves latent-X + scaled-Y + removed to:
      data/sets/scaled_sets/df_scaled_x_flowpre_{meta_tag}_y{y_scaler_name}/
    Also saves the Y scaler AND copies the model's YAML + .pt into ./model/.
    """
    dirs = _flowpre_dirs(meta_tag, y_scaler_name)
    _ensure_dirs(dirs)
    suffix = _flowpre_suffix(meta_tag, y_scaler_name)

    # --- Fit Y scaler on TRAIN (exclude ids and condition)
    exclude_cols = ["post_cleaning_index", condition_col]
    y_cols = [c for c in y_train.columns if c not in exclude_cols]
    scaler = get_scaler(y_scaler_name).fit(y_train[y_cols])

    # Transform Y splits
    y_train_sc = y_train.copy(); y_train_sc[y_cols] = scaler.transform(y_train[y_cols])
    y_val_sc   = y_val.copy();   y_val_sc[y_cols]   = scaler.transform(y_val[y_cols])
    y_test_sc  = y_test.copy();  y_test_sc[y_cols]  = scaler.transform(y_test[y_cols])

    # Save CSVs
    X_lat_train.to_csv(dirs["X"] / f"{suffix}_X_train.csv", index=False)
    X_lat_val.to_csv(  dirs["X"] / f"{suffix}_X_val.csv",   index=False)
    X_lat_test.to_csv( dirs["X"] / f"{suffix}_X_test.csv",  index=False)

    y_train_sc.to_csv(dirs["y"] / f"{suffix}_y_train.csv", index=False)
    y_val_sc.to_csv(  dirs["y"] / f"{suffix}_y_val.csv",   index=False)
    y_test_sc.to_csv( dirs["y"] / f"{suffix}_y_test.csv",  index=False)

    r_train.to_csv(dirs["removed"] / f"{suffix}_r_train.csv", index=False)
    r_val.to_csv(  dirs["removed"] / f"{suffix}_r_val.csv",   index=False)
    r_test.to_csv( dirs["removed"] / f"{suffix}_r_test.csv",  index=False)

    # Save Y scaler
    scaler_path = dirs["scalers"] / f"{suffix}_y_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Copy model YAML + .pt into ./model/
    copied = _copy_model_artifacts(model_name, dirs["model"])

    # Manifest
    manifest = {
        "x_transform": {
            "type": "flowpre_latent",
            "model_name": model_name,
            "source_dir": copied["source_dir"],
            "copied_config": copied["config"],
            "copied_weights": copied["weights"],
        },
        "y_scaler": {
            "name": y_scaler_name,
            "path": str(scaler_path),
            "fit_cols": y_cols,
        },
        "condition_col": condition_col,
        "suffix": suffix,
    }
    (dirs["meta"]).mkdir(parents=True, exist_ok=True)
    with open(dirs["meta"] / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        rel = dirs["base"].relative_to(Path(ROOT_PATH))
        log(f"✅ Saved FlowPre set → {rel}\n    ↳ + config/weights copied to {dirs['model'].relative_to(Path(ROOT_PATH))}")
    return dirs["base"], manifest

def load_flowpre_scaled_set(
    meta_tag: str,
    y_scaler_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           object]:
    """
    Loads the nine CSVs + the saved Y scaler for a given FlowPre set.
    Returns:
      X_train, X_val, X_test,
      y_train, y_val, y_test,
      r_train, r_val, r_test,
      y_scaler
    """
    dirs = _flowpre_dirs(meta_tag, y_scaler_name)
    suffix = _flowpre_suffix(meta_tag, y_scaler_name)

    X_train = pd.read_csv(dirs["X"] / f"{suffix}_X_train.csv")
    X_val   = pd.read_csv(dirs["X"] / f"{suffix}_X_val.csv")
    X_test  = pd.read_csv(dirs["X"] / f"{suffix}_X_test.csv")

    y_train = pd.read_csv(dirs["y"] / f"{suffix}_y_train.csv")
    y_val   = pd.read_csv(dirs["y"] / f"{suffix}_y_val.csv")
    y_test  = pd.read_csv(dirs["y"] / f"{suffix}_y_test.csv")

    r_train = pd.read_csv(dirs["removed"] / f"{suffix}_r_train.csv")
    r_val   = pd.read_csv(dirs["removed"] / f"{suffix}_r_val.csv")
    r_test  = pd.read_csv(dirs["removed"] / f"{suffix}_r_test.csv")

    y_scaler = joblib.load(dirs["scalers"] / f"{suffix}_y_scaler.pkl")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            r_train, r_val, r_test,
            y_scaler)

def load_or_create_flowpre_sets(
    model_map: Dict[str, str],              # {"rrmse": "<model_folder>", "mvn": "...", "fair": "..."}
    y_scalers: List[str],                   # e.g., ["minmax","quantile","standard"]
    df_name: str = "df_input",
    condition_col: str = "type",
    val_size: int = 150,
    test_size: int = 100,
    target: str = "init",
    device: str = "cuda",
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    For each FlowPre model (tag → folder), encodes X into latent z, then for each y_scaler,
    creates a new set under:
      data/sets/scaled_sets/df_scaled_x_flowpre_{tag}_y{scaler}
    Also ensures the model's YAML + .pt are copied into each set's ./model/ dir.
    Returns a list of created/loaded set suffixes.
    """
    # 1) raw splits once
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     r_train, r_val, r_test) = load_or_create_raw_splits(
        df_name=df_name,
        condition_col=condition_col,
        val_size=val_size,
        test_size=test_size,
        target=target,
        force=False,
        verbose=verbose
    )

    created = []
    for tag, model_name in model_map.items():
        if verbose:
            log(f"🌀 Encoding X with FlowPre model [{tag}] → {model_name}")
        # 2) X → latent with chosen FlowPre
        X_lat_train, X_lat_val, X_lat_test = transform_X_with_flowpre_model(
            model_name=model_name,
            X_train=X_train, X_val=X_val, X_test=X_test,
            condition_col=condition_col,
            cols_to_exclude=["post_cleaning_index"],
            device=device,
            verbose=verbose
        )

        # 3) per-Y-scaler save bundle (+ copy model artifacts)
        for y_s in y_scalers:
            dirs = _flowpre_dirs(tag, y_s)
            suffix = _flowpre_suffix(tag, y_s)

            if not force and _set_has_all_core_files(dirs, suffix):
                # Even if skipping CSV/scaler creation, ensure artifacts exist (idempotent)
                _copy_model_artifacts(model_name, dirs["model"])
                if verbose:
                    rel = dirs["base"].relative_to(Path(ROOT_PATH))
                    log(f"📦 Exists → {rel} (verified config/weights present)")
                created.append(suffix)
                continue

            base_dir, _ = save_flowpre_scaled_bundle(
                X_lat_train, X_lat_val, X_lat_test,
                y_train, y_val, y_test,
                r_train, r_val, r_test,
                y_scaler_name=y_s,
                meta_tag=tag,
                model_name=model_name,
                condition_col=condition_col,
                verbose=verbose
            )
            created.append(base_dir.name)

    return created

