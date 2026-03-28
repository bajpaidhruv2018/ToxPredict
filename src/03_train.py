# ============================================
# STEP 3: TRAIN MODELS
# ============================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("LOADING PROCESSED DATA")
print("=" * 50)

df = pd.read_csv('data/tox21_processed.csv')
print(f"✅ Loaded data: {df.shape}")

# Define feature columns
basic_cols = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
    'NumHeavyAtoms', 'FractionCSP3'
]

fp_cols  = [f'FP_{i}' for i in range(1024)]
tox_cols = [c for c in df.columns if c.startswith('tox_')]

# Use all features combined
feature_cols = basic_cols + fp_cols + tox_cols
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"✅ Total features being used: {len(feature_cols)}")

# Toxicity targets
tox_targets = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
tox_targets = [t for t in tox_targets if t in df.columns]

print(f"✅ Training for {len(tox_targets)} toxicity targets")
print("\n" + "=" * 50)
print("TRAINING MODELS")
print("=" * 50)

results = {}
models  = {}

for target in tox_targets:
    print(f"\n  Training: {target}")

    # Drop rows where target is missing
    df_t = df[feature_cols + [target]].dropna()

    X = df_t[feature_cols]
    y = df_t[target]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"    Class ratio (imbalance): {(y_train==0).sum()}:{(y_train==1).sum()}")

    # Handle class imbalance
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Define 3 models
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

    # Ensemble — combines all 3
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf',  rf),
            ('lr',  lr)
        ],
        voting='soft'
    )

    # Train
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred  = ensemble.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)

    results[target] = {
        'ROC-AUC': round(auc, 4),
        'F1':      round(f1, 4),
        'Train_size': len(X_train),
        'Test_size':  len(X_test),
    }
    models[target] = {
        'model':        ensemble,
        'feature_cols': feature_cols
    }

    print(f"    ✅ AUC: {auc:.4f} | F1: {f1:.4f}")

# Save everything
print("\n" + "=" * 50)
print("SAVING RESULTS")
print("=" * 50)

with open('results/models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✅ Models saved to results/models.pkl")

results_df = pd.DataFrame(results).T
results_df.to_csv('results/metrics.csv')
print("✅ Metrics saved to results/metrics.csv")

print("\n📊 FINAL RESULTS:")
print(results_df.to_string())
print("\n🎉 Training Complete!")