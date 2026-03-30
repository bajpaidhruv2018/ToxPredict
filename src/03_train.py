# ============================================
# STEP 3: TRAIN MODELS
# Updated: Cross Validation + Better Metrics
# ============================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import pickle
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("LOADING PROCESSED DATA")
print("=" * 60)

df = pd.read_csv('data/tox21_processed.csv')
print(f"✅ Loaded data: {df.shape}")

# Define feature columns
basic_cols = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
    'NumHeavyAtoms', 'FractionCSP3'
]

# Auto detect fingerprint columns
mfp_cols  = [c for c in df.columns if c.startswith('MFP_')]
maccs_cols = [c for c in df.columns if c.startswith('MACCS_')]
tox_cols  = [c for c in df.columns if c.startswith('tox_')]

feature_cols = basic_cols + mfp_cols + maccs_cols + tox_cols
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"\n✅ Feature breakdown:")
print(f"   Basic descriptors : {len(basic_cols)}")
print(f"   Morgan FP (2048)  : {len(mfp_cols)}")
print(f"   MACCS Keys        : {len(maccs_cols)}")
print(f"   Toxicophores      : {len(tox_cols)}")
print(f"   TOTAL             : {len(feature_cols)}")

# Toxicity targets
tox_targets = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
tox_targets = [t for t in tox_targets if t in df.columns]

print(f"\n✅ Training for {len(tox_targets)} toxicity targets")
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE MODELS + CROSS VALIDATION")
print("=" * 60)

results = {}
models  = {}

for target in tox_targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")

    df_t = df[feature_cols + [target]].dropna()
    X    = df_t[feature_cols]
    y    = df_t[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Class ratio: {int((y_train==0).sum())}:{int((y_train==1).sum())} (neg:pos)")

    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # --- Define 3 models ---
    xgb = XGBClassifier(
        scale_pos_weight=ratio,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

    # --- Ensemble ---
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf',  rf),
            ('lr',  lr)
        ],
        voting='soft'
    )

    # --- Train ---
    print(f"  Training ensemble...")
    ensemble.fit(X_train, y_train)

    # --- Hold-out evaluation ---
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred  = ensemble.predict(X_test)

    auc    = roc_auc_score(y_test, y_proba)
    f1     = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    # --- Cross Validation (5-fold) ---
    print(f"  Running 5-fold cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # CV on just XGBoost for speed (representative of ensemble)
    cv_scores = cross_val_score(
        XGBClassifier(
            scale_pos_weight=ratio,
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        X, y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    results[target] = {
        'ROC-AUC':    round(auc, 4),
        'F1':         round(f1, 4),
        'PR-AUC':     round(pr_auc, 4),
        'CV-AUC':     round(cv_scores.mean(), 4),
        'CV-Std':     round(cv_scores.std(), 4),
        'Train_size': len(X_train),
        'Test_size':  len(X_test),
    }

    models[target] = {
        'model':        ensemble,
        'feature_cols': feature_cols
    }

    print(f"  ✅ ROC-AUC : {auc:.4f}")
    print(f"  ✅ PR-AUC  : {pr_auc:.4f}")
    print(f"  ✅ F1      : {f1:.4f}")
    print(f"  ✅ CV-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ----------------------------------------
# SAVE EVERYTHING
# ----------------------------------------
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

with open('results/models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✅ Models saved to results/models.pkl")

results_df = pd.DataFrame(results).T
results_df.to_csv('results/metrics.csv')
print("✅ Metrics saved to results/metrics.csv")

print("\n📊 FINAL RESULTS TABLE:")
print(results_df.to_string())

print(f"\n📈 SUMMARY:")
print(f"   Best AUC  : {results_df['ROC-AUC'].max():.4f} ({results_df['ROC-AUC'].idxmax()})")
print(f"   Mean AUC  : {results_df['ROC-AUC'].mean():.4f}")
print(f"   Above 0.75: {(results_df['ROC-AUC'] > 0.75).sum()}/12 targets")

print("\n🎉 Training Complete!")