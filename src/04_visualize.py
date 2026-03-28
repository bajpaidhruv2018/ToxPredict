# ============================================
# STEP 4: GENERATE ALL VISUALIZATIONS
# ============================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("LOADING DATA AND MODELS")
print("=" * 50)

df         = pd.read_csv('data/tox21_processed.csv')
results_df = pd.read_csv('results/metrics.csv', index_col=0)

with open('results/models.pkl', 'rb') as f:
    models = pickle.load(f)

tox_targets  = list(results_df.index)
basic_cols   = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
    'NumHeavyAtoms', 'FractionCSP3'
]
tox_cols     = [c for c in df.columns if c.startswith('tox_')]
feature_cols = basic_cols + tox_cols

print("✅ Everything loaded")

# ----------------------------------------
# PLOT 1: Per-Assay AUC Bar Chart
# ----------------------------------------
print("\nGenerating Plot 1: AUC Bar Chart...")

plt.figure(figsize=(12, 5))
colors = ['#2ecc71' if x >= 0.75 else '#e74c3c'
          for x in results_df['ROC-AUC']]

bars = plt.bar(results_df.index, results_df['ROC-AUC'], color=colors)
plt.axhline(y=0.75, color='black', linestyle='--',
            linewidth=1.5, label='0.75 threshold')

# Add value labels on bars
for bar, val in zip(bars, results_df['ROC-AUC']):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{val:.3f}',
        ha='center', va='bottom', fontsize=8
    )

plt.xticks(rotation=45, ha='right')
plt.ylabel("ROC-AUC Score")
plt.ylim(0, 1.0)
plt.title("Model Performance Across All 12 Tox21 Assays", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('results/01_per_assay_auc.png', dpi=150)
plt.close()
print("  ✅ Saved 01_per_assay_auc.png")

# ----------------------------------------
# PLOT 2: Correlation Heatmap
# ----------------------------------------
print("\nGenerating Plot 2: Correlation Heatmap...")

corr_data   = df[basic_cols + tox_targets].corr()
corr_subset = corr_data.loc[basic_cols, tox_targets]

plt.figure(figsize=(14, 6))
sns.heatmap(
    corr_subset,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation"}
)
plt.title("Molecular Properties vs Toxicity Targets", fontsize=13)
plt.tight_layout()
plt.savefig('results/02_correlation_heatmap.png', dpi=150)
plt.close()
print("  ✅ Saved 02_correlation_heatmap.png")

# ----------------------------------------
# PLOT 3: Toxicophore Analysis
# ----------------------------------------
print("\nGenerating Plot 3: Toxicophore Analysis...")

tox_counts = df[tox_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(
    [c.replace('tox_', '').replace('_', ' ') for c in tox_counts.index],
    tox_counts.values,
    color='#e67e22'
)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Molecules")
plt.title("Toxicophore Frequency in Tox21 Dataset", fontsize=13)
plt.tight_layout()
plt.savefig('results/03_toxicophore_frequency.png', dpi=150)
plt.close()
print("  ✅ Saved 03_toxicophore_frequency.png")

# ----------------------------------------
# PLOT 4: SHAP Summary Plot
# ----------------------------------------
print("\nGenerating Plot 4: SHAP Summary...")

# Pick best performing assay
best_target = results_df['ROC-AUC'].idxmax()
print(f"  Best assay: {best_target} (AUC={results_df.loc[best_target,'ROC-AUC']})")

model_data   = models[best_target]
ensemble     = model_data['model']
feat_cols    = model_data['feature_cols']

df_t = df[feat_cols + [best_target]].dropna()
X    = df_t[feat_cols]

# Use XGBoost from ensemble for SHAP
rf_model  = ensemble.named_estimators_['rf']
explainer = shap.TreeExplainer(rf_model)

# Use small sample for speed
X_sample    = X.sample(min(500, len(X)), random_state=42)
shap_values = explainer.shap_values(X_sample)

# Only show basic + toxicophore features (not all 1024 FPs)
show_features = basic_cols + tox_cols
show_indices  = [feat_cols.index(f) for f in show_features if f in feat_cols]

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values[:, show_indices],
    X_sample[show_features],
    feature_names=show_features,
    show=False,
    plot_type='beeswarm'
)
plt.title(f"SHAP Feature Importance — {best_target}", fontsize=13)
plt.tight_layout()
plt.savefig('results/04_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 04_shap_summary.png")

# ----------------------------------------
# PLOT 5: Real Drug Predictions
# ----------------------------------------
print("\nGenerating Plot 5: Real Drug Predictions...")

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MolFromSmarts

TOXICOPHORES = {
    'tox_Nitro_group':      '[N+](=O)[O-]',
    'tox_Aldehyde':         '[CH]=O',
    'tox_Epoxide':          'C1OC1',
    'tox_Aromatic_amine':   'Nc1ccccc1',
    'tox_Hydrazine':        'NN',
    'tox_Alkyl_halide':     '[CX4][F,Cl,Br,I]',
    'tox_Michael_acceptor': 'C=CC=O',
    'tox_Quinone':          'O=C1C=CC(=O)C=C1',
    'tox_Azo_compound':     'N=N',
    'tox_Peroxide':         'OO',
}

def predict_molecule(smiles, models, feat_cols):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    tox_flags = {}
    for name, smarts in TOXICOPHORES.items():
        pattern = MolFromSmarts(smarts)
        tox_flags[name] = int(mol.HasSubstructMatch(pattern)) if pattern else 0
    tox_flags['total_toxicophores'] = sum(tox_flags.values())

    basic = {
        'MolWt':             Descriptors.MolWt(mol),
        'LogP':              Descriptors.MolLogP(mol),
        'NumHDonors':        rdMolDescriptors.CalcNumHBD(mol),
        'NumHAcceptors':     rdMolDescriptors.CalcNumHBA(mol),
        'TPSA':              Descriptors.TPSA(mol),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'NumAromaticRings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'NumHeavyAtoms':     mol.GetNumHeavyAtoms(),
        'FractionCSP3':      rdMolDescriptors.CalcFractionCSP3(mol),
    }

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_dict = {f'FP_{i}': int(fp[i]) for i in range(1024)}

    all_feats = {**basic, **fp_dict, **tox_flags}
    row       = pd.DataFrame([all_feats])
    row       = row.reindex(columns=feat_cols, fill_value=0)

    predictions = {}
    for target, model_data in models.items():
        prob = model_data['model'].predict_proba(row)[0][1]
        predictions[target] = round(prob, 3)

    return predictions

real_drugs = {
    'Aspirin':     'CC(=O)Oc1ccccc1C(=O)O',
    'Ibuprofen':   'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'Caffeine':    'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
}

feat_cols_main = models[list(models.keys())[0]]['feature_cols']
drug_results   = {}

for drug_name, smiles in real_drugs.items():
    preds = predict_molecule(smiles, models, feat_cols_main)
    if preds:
        drug_results[drug_name] = preds
        print(f"  ✅ Predicted: {drug_name}")

drug_df = pd.DataFrame(drug_results).T
drug_df.to_csv('results/real_drug_predictions.csv')

plt.figure(figsize=(12, 5))
sns.heatmap(
    drug_df,
    annot=True,
    fmt=".2f",
    cmap='RdYlGn_r',
    vmin=0, vmax=1,
    linewidths=0.5
)
plt.title("Toxicity Risk Across Real FDA Drugs", fontsize=13)
plt.tight_layout()
plt.savefig('results/05_real_drug_predictions.png', dpi=150)
plt.close()
print("  ✅ Saved 05_real_drug_predictions.png")

print("\n" + "=" * 50)
print("ALL VISUALIZATIONS DONE")
print("=" * 50)
print("\nFiles saved in results/:")
for f in sorted(os.listdir('results')):
    print(f"  ✅ {f}")
print("\n🎉 Visualization Complete!")