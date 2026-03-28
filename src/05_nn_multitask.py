# ============================================
# STEP 5: MULTI-TASK NEURAL NETWORK (PyTorch)
# ============================================
# A feed-forward MLP that predicts all 12 Tox21
# toxicity targets simultaneously. Uses a boolean
# mask to handle pervasive NaN labels so that
# missing assays contribute zero gradient.
# ============================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────

TRAIN_PATH   = "data/tox21_train_processed.csv"
TEST_PATH    = "data/tox21_test_processed.csv"
MODEL_PATH   = "results/multitask_mlp.pth"
BATCH_SIZE   = 256
EPOCHS       = 20
LR           = 1e-3
DROPOUT      = 0.3
RANDOM_SEED  = 42

# The 12 Tox21 toxicity assay targets
TOX_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ────────────────────────────────────────────
# 1. DATASET WITH NaN MASKING
# ────────────────────────────────────────────

class Tox21MultiTaskDataset(Dataset):
    """
    Custom Dataset that returns (features, labels, mask).

    • mask[i] = 1.0 where the label is valid
    • mask[i] = 0.0 where the label was NaN
    • NaN labels are filled with 0.0 so PyTorch
      tensors remain valid (but masked out in loss).
    """

    def __init__(self, csv_path: str, target_cols: list):
        df = pd.read_csv(csv_path)

        # Separate targets from features
        # Keep only targets that actually exist in the file
        self.target_cols = [c for c in target_cols if c in df.columns]
        y = df[self.target_cols].values.astype(np.float32)

        # Build the mask BEFORE filling NaNs
        self.mask = (~np.isnan(y)).astype(np.float32)      # 1 = valid, 0 = missing
        y = np.nan_to_num(y, nan=0.0)                      # fill NaN → 0

        # Everything that is NOT a target and IS numeric is a feature
        non_feature_cols = set(self.target_cols)
        feature_cols = [c for c in df.columns
                        if c not in non_feature_cols
                        and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols].values.astype(np.float32)

        # Convert to tensors
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.mask = torch.tensor(self.mask)
        self.n_features = X.shape[1]

        print(f"  ✅ Loaded {csv_path}")
        print(f"     Samples : {len(self.X)}")
        print(f"     Features: {self.n_features}")
        print(f"     Targets : {len(self.target_cols)}")
        print(f"     Valid labels: {int(self.mask.sum())} / {self.mask.numel()} "
              f"({self.mask.sum() / self.mask.numel() * 100:.1f}%)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


# ────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ────────────────────────────────────────────

class ToxMultiTaskMLP(nn.Module):
    """
    Feed-forward multi-task network.

    Input(n_features)
      → Linear(512) → ReLU → BatchNorm → Dropout(0.3)
      → Linear(256) → ReLU → BatchNorm → Dropout(0.3)
      → Linear(12)   ← raw logits (no Sigmoid; BCEWithLogitsLoss)
    """

    def __init__(self, n_features: int, n_targets: int = 12, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            # Block 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            # Output — raw logits
            nn.Linear(256, n_targets),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────
# 3. MASKED TRAINING LOOP
# ────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch and return mean masked loss."""
    model.train()
    total_loss = 0.0
    total_valid = 0

    for X_batch, y_batch, mask_batch in loader:
        X_batch  = X_batch.to(device)
        y_batch  = y_batch.to(device)
        mask_batch = mask_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)                         # (B, 12)
        raw_loss = criterion(logits, y_batch)            # (B, 12)  — reduction='none'

        # Zero-out loss for missing labels, average over valid entries
        n_valid = mask_batch.sum()
        if n_valid > 0:
            masked_loss = (raw_loss * mask_batch).sum() / n_valid
        else:
            masked_loss = torch.tensor(0.0, device=device)

        masked_loss.backward()
        optimizer.step()

        total_loss  += masked_loss.item() * n_valid.item()
        total_valid += n_valid.item()

    return total_loss / max(total_valid, 1)


# ────────────────────────────────────────────
# 4. EVALUATION
# ────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, target_cols):
    """
    Run model on a DataLoader and compute per-target
    ROC-AUC (only where labels are valid).
    Returns dict of {target: auc}.
    """
    model.eval()
    all_logits, all_labels, all_masks = [], [], []

    for X_batch, y_batch, mask_batch in loader:
        logits = model(X_batch.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y_batch)
        all_masks.append(mask_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    masks  = torch.cat(all_masks).numpy()

    # Sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    results = {}
    for i, col in enumerate(target_cols):
        valid = masks[:, i] == 1.0
        if valid.sum() < 10:
            results[col] = float("nan")
            continue
        y_true = labels[valid, i]
        y_prob = probs[valid, i]
        # Need both classes present
        if len(np.unique(y_true)) < 2:
            results[col] = float("nan")
            continue
        results[col] = roc_auc_score(y_true, y_prob)

    return results, probs


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: MULTI-TASK NEURAL NETWORK (PyTorch)")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # --- Data ---
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)

    train_ds = Tox21MultiTaskDataset(TRAIN_PATH, TOX_TARGETS)
    test_ds  = Tox21MultiTaskDataset(TEST_PATH,  TOX_TARGETS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    target_cols = train_ds.target_cols

    # --- Model ---
    print("\n" + "-" * 60)
    print("MODEL ARCHITECTURE")
    print("-" * 60)

    model = ToxMultiTaskMLP(
        n_features=train_ds.n_features,
        n_targets=len(target_cols),
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\n  Total parameters: {total_params:,}")

    # --- Training ---
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Quick test-set AUC every 5 epochs and on the last epoch
        if epoch % 5 == 0 or epoch == EPOCHS:
            test_aucs, _ = evaluate(model, test_loader, device, target_cols)
            mean_auc = np.nanmean(list(test_aucs.values()))
            print(f"  Epoch {epoch:>2}/{EPOCHS}  |  "
                  f"Train Loss: {train_loss:.4f}  |  "
                  f"Test Mean AUC: {mean_auc:.4f}")
        else:
            print(f"  Epoch {epoch:>2}/{EPOCHS}  |  "
                  f"Train Loss: {train_loss:.4f}")

    # --- Final evaluation ---
    print("\n" + "-" * 60)
    print("FINAL TEST-SET EVALUATION")
    print("-" * 60)

    test_aucs, test_probs = evaluate(model, test_loader, device, target_cols)

    print(f"\n  {'Target':<18} {'ROC-AUC':>8}")
    print("  " + "-" * 28)
    for col, auc in test_aucs.items():
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A"
        print(f"  {col:<18} {auc_str:>8}")

    valid_aucs = [v for v in test_aucs.values() if not np.isnan(v)]
    print("  " + "-" * 28)
    print(f"  {'Mean AUC':<18} {np.mean(valid_aucs):>8.4f}")

    # --- Save ---
    print("\n" + "-" * 60)
    print("SAVING MODEL")
    print("-" * 60)

    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": train_ds.n_features,
        "n_targets": len(target_cols),
        "target_cols": target_cols,
        "dropout": DROPOUT,
    }, MODEL_PATH)
    print(f"  ✅ Model weights saved to {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("🎉 MULTI-TASK NEURAL NETWORK TRAINING COMPLETE")
    print("=" * 60)
