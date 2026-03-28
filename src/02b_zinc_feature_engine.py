# ============================================
# STEP 2B: ZINC250k SURROGATE QED FEATURE ENGINE
# ============================================
# Trains a LightGBM surrogate on ZINC250k to
# predict QED from Morgan fingerprints, then
# enriches the Tox21 processed datasets with
# the predicted "Surrogate_QED" feature.
# ============================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, QED

# ────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────

ZINC_PATH          = 'data/zinc250k.csv'
ZINC_SAMPLE_SIZE   = 50_000
FP_BITS            = 1024
FP_RADIUS          = 2
SURROGATE_PATH     = 'results/zinc_qed_surrogate.pkl'
TRAIN_PROCESSED    = 'data/tox21_train_processed.csv'
TEST_PROCESSED     = 'data/tox21_test_processed.csv'
RANDOM_SEED        = 42

# Public URL for ZINC250k (commonly used benchmark CSV)
ZINC_URL = (
    'https://raw.githubusercontent.com/aspuru-guzik-group'
    '/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
)

os.makedirs('results', exist_ok=True)


# ────────────────────────────────────────────
# 1. ZINC DATA HANDLING
# ────────────────────────────────────────────

def download_zinc(url: str, dest: str) -> None:
    """Download ZINC250k CSV from a public URL."""
    print(f"  ⬇️  Downloading ZINC250k from:\n      {url}")
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  ✅ Downloaded to {dest}")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print("  👉 Please manually place a CSV with a 'smiles' column at:", dest)
        sys.exit(1)


def load_zinc(path: str, url: str) -> pd.DataFrame:
    """Load ZINC250k CSV, downloading if necessary."""
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found locally.")
        download_zinc(url, path)

    df = pd.read_csv(path)
    print(f"  ✅ Loaded ZINC250k: {len(df)} molecules")

    # Normalise column name — the public CSV uses 'smiles' or 'SMILES'
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    if 'smiles' not in df.columns:
        raise ValueError(
            f"Expected a 'smiles' column in {path}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def compute_zinc_features(df: pd.DataFrame,
                           sample_size: int,
                           seed: int) -> tuple:
    """
    Sample molecules, compute Morgan FPs and QED scores.
    Returns (fingerprint_array, qed_array).
    """
    n = min(sample_size, len(df))
    sampled = df.sample(n=n, random_state=seed)
    print(f"\n  🎲 Sampled {n} molecules from ZINC250k")

    fps, qeds = [], []
    skipped = 0

    for i, smiles in enumerate(sampled['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FP_RADIUS, nBits=FP_BITS
        )
        fps.append(np.array(fp))
        qeds.append(QED.qed(mol))

        if (i + 1) % 10_000 == 0:
            print(f"    Progress: {i + 1}/{n}")

    print(f"  ✅ Computed features for {len(fps)} molecules "
          f"(skipped {skipped} invalid SMILES)")

    return np.array(fps), np.array(qeds)


# ────────────────────────────────────────────
# 2. SURROGATE MODEL TRAINING
# ────────────────────────────────────────────

def train_surrogate(X: np.ndarray, y: np.ndarray, save_path: str):
    """Train a LightGBM regressor to predict QED from fingerprints."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  ❌ lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    print(f"\n  🏋️  Training LGBMRegressor on {X.shape[0]} samples, "
          f"{X.shape[1]} features...")

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X, y)

    # Quick in-sample sanity check
    preds = model.predict(X)
    mae = np.mean(np.abs(preds - y))
    print(f"  ✅ Training MAE (in-sample): {mae:.4f}")
    print(f"  ✅ QED range in training data: [{y.min():.4f}, {y.max():.4f}]")

    joblib.dump(model, save_path)
    print(f"  ✅ Surrogate model saved to {save_path}")

    return model


# ────────────────────────────────────────────
# 3. TOX21 ENRICHMENT
# ────────────────────────────────────────────

def enrich_tox21(csv_path: str, model, fp_cols: list, label: str):
    """
    Load a Tox21 processed CSV, predict Surrogate_QED
    using the ZINC-trained surrogate, and overwrite the file.
    """
    df = pd.read_csv(csv_path)
    print(f"\n  📂 Loaded {label}: {df.shape}")

    # Extract existing fingerprint columns
    available_fp = [c for c in fp_cols if c in df.columns]
    if len(available_fp) != FP_BITS:
        print(f"  ⚠️  Expected {FP_BITS} FP columns, found {len(available_fp)}. "
              f"Proceeding with {len(available_fp)}.")

    X = df[available_fp].values

    # Predict Surrogate QED
    surrogate_qed = model.predict(X)

    # Clip to valid QED range [0, 1]
    surrogate_qed = np.clip(surrogate_qed, 0.0, 1.0)

    df['Surrogate_QED'] = surrogate_qed

    print(f"  ✅ Surrogate_QED stats for {label}:")
    print(f"     mean={surrogate_qed.mean():.4f}  "
          f"std={surrogate_qed.std():.4f}  "
          f"min={surrogate_qed.min():.4f}  "
          f"max={surrogate_qed.max():.4f}")

    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved enriched {label} → {csv_path}  "
          f"(new shape: {df.shape})")


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 2B: ZINC250k SURROGATE QED FEATURE ENGINE")
    print("=" * 60)

    # --- Phase 1: Load & sample ZINC ---
    print("\n" + "-" * 60)
    print("PHASE 1: ZINC DATA LOADING & SAMPLING")
    print("-" * 60)
    zinc_df = load_zinc(ZINC_PATH, ZINC_URL)
    X_zinc, y_qed = compute_zinc_features(
        zinc_df, ZINC_SAMPLE_SIZE, RANDOM_SEED
    )

    # --- Phase 2: Train surrogate ---
    print("\n" + "-" * 60)
    print("PHASE 2: SURROGATE MODEL TRAINING")
    print("-" * 60)
    surrogate = train_surrogate(X_zinc, y_qed, SURROGATE_PATH)

    # --- Phase 3: Enrich Tox21 ---
    print("\n" + "-" * 60)
    print("PHASE 3: TOX21 ENRICHMENT")
    print("-" * 60)
    fp_cols = [f'FP_{i}' for i in range(FP_BITS)]

    enrich_tox21(TRAIN_PROCESSED, surrogate, fp_cols, label='Tox21 Train')
    enrich_tox21(TEST_PROCESSED,  surrogate, fp_cols, label='Tox21 Test')

    # --- Done ---
    print("\n" + "=" * 60)
    print("🎉 ZINC SURROGATE QED FEATURE ENGINE COMPLETE")
    print("=" * 60)
    print(f"  • Surrogate model : {SURROGATE_PATH}")
    print(f"  • Enriched train  : {TRAIN_PROCESSED}")
    print(f"  • Enriched test   : {TEST_PROCESSED}")
    print("  • New column      : Surrogate_QED")
    print("=" * 60)
