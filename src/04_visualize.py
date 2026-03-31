# ============================================
# STEP 4: GENERATE ALL VISUALIZATIONS
# Updated: ROC curves + all previous plots
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MolFromSmarts

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("LOADING DATA AND MODELS")
print("=" * 60)

df         = pd.read_csv('data/tox21_processed.csv')
results_df = pd.read_csv('results/metrics.csv', index_col=0)

with open('results/models.pkl', 'rb') as f:
    models = pickle.load(f)

tox_targets = list(results_df.index)

basic_cols  = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
    'NumHeavyAtoms', 'FractionCSP3'
]
tox_cols    = [c for c in df.columns if c.startswith('tox_')]
feature_cols = basic_cols + tox_cols

print(f"✅ Loaded {len(df)} molecules")
print(f"✅ {len(models)} trained models")
print(f"✅ Generating {len(tox_targets)} target plots")

# ----------------------------------------
# PLOT 1: Per-Assay AUC Bar Chart
# ----------------------------------------
print("\n[1/6] Generating AUC Bar Chart...")

plt.figure(figsize=(14, 6))
colors = ['#2ecc71' if x >= 0.75 else '#e74c3c'
          for x in results_df['ROC-AUC']]

bars = plt.bar(results_df.index, results_df['ROC-AUC'], color=colors, edgecolor='white', linewidth=0.5)
plt.axhline(y=0.75, color='white', linestyle='--', linewidth=1.5, label='0.75 industry threshold', alpha=0.7)
plt.axhline(y=0.80, color='#f39c12', linestyle=':', linewidth=1.5, label='0.80 state-of-the-art', alpha=0.7)

for bar, val in zip(bars, results_df['ROC-AUC']):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{val:.3f}',
        ha='center', va='bottom', fontsize=8, color='white'
    )

plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.ylabel("ROC-AUC Score", color='white')
plt.ylim(0, 1.05)
plt.title("ToxPredict — Model Performance Across All 12 Tox21 Assays", fontsize=13, color='white')
plt.legend(fontsize=10)
plt.gca().set_facecolor('#161b22')
plt.gcf().set_facecolor('#0d1117')
plt.tight_layout()
plt.savefig('results/01_per_assay_auc.png', dpi=150, facecolor='#0d1117')
plt.close()
print("  ✅ Saved 01_per_assay_auc.png")

# ----------------------------------------
# PLOT 2: Correlation Heatmap
# ----------------------------------------
print("\n[2/6] Generating Correlation Heatmap...")

corr_data   = df[basic_cols + tox_targets].corr()
corr_subset = corr_data.loc[basic_cols, tox_targets]

plt.figure(figsize=(16, 7))
sns.heatmap(
    corr_subset,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"},
    annot_kws={"size": 9}
)
plt.title("Molecular Properties vs Toxicity Targets\n(Pearson Correlation)", fontsize=13)
plt.tight_layout()
plt.savefig('results/02_correlation_heatmap.png', dpi=150)
plt.close()
print("  ✅ Saved 02_correlation_heatmap.png")

# ----------------------------------------
# PLOT 3: Toxicophore Frequency
# ----------------------------------------
print("\n[3/6] Generating Toxicophore Analysis...")

tox_counts = df[tox_cols].sum().sort_values(ascending=False)
labels     = [c.replace('tox_', '').replace('_', ' ') for c in tox_counts.index]

plt.figure(figsize=(14, 6))
bars = plt.bar(labels, tox_counts.values, color='#e67e22', edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, tox_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 5,
        str(int(val)),
        ha='center', va='bottom', fontsize=9
    )

plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Molecules")
plt.title("Toxicophore Frequency in Tox21 Dataset\n(Known Toxic Substructures)", fontsize=13)
plt.tight_layout()
plt.savefig('results/03_toxicophore_frequency.png', dpi=150)
plt.close()
print("  ✅ Saved 03_toxicophore_frequency.png")

# ----------------------------------------
# PLOT 4: SHAP Summary Plot
# ----------------------------------------
print("\n[4/6] Generating SHAP Summary...")

best_target = results_df['ROC-AUC'].idxmax()
print(f"  Best assay: {best_target} (AUC={results_df.loc[best_target,'ROC-AUC']})")

model_data = models[best_target]
ensemble   = model_data['model']
feat_cols  = model_data['feature_cols']

df_t = df[feat_cols + [best_target]].dropna()
X    = df_t[feat_cols]

# Use Random Forest for SHAP (avoids XGBoost version conflict)
rf_model    = ensemble.named_estimators_['rf']
explainer   = shap.TreeExplainer(rf_model)
X_sample    = X.sample(min(300, len(X)), random_state=42)
shap_values = explainer.shap_values(X_sample)

# Show only interpretable features (not all fingerprints)
show_features = basic_cols + tox_cols
show_features = [f for f in show_features if f in feat_cols]
show_indices  = [feat_cols.index(f) for f in show_features]

if isinstance(shap_values, list):
    sv = shap_values[1][:, show_indices]
else:
    sv = shap_values[:, show_indices]

plt.figure(figsize=(10, 8))
shap.summary_plot(
    sv,
    X_sample[show_features],
    feature_names=[f.replace('tox_', '⚠ ').replace('_', ' ') for f in show_features],
    show=False,
    plot_type='beeswarm'
)
plt.title(f"SHAP Feature Importance — {best_target}", fontsize=13)
plt.tight_layout()
plt.savefig('results/04_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 04_shap_summary.png")

# ----------------------------------------
# PLOT 5: Real Drug Predictions Heatmap
# ----------------------------------------
print("\n[5/6] Generating Real Drug Predictions...")

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

def predict_molecule(smiles, models):
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

    feat_cols = models[list(models.keys())[0]]['feature_cols']
    mfp_size  = len([c for c in feat_cols if c.startswith('MFP_')])
    maccs_size = len([c for c in feat_cols if c.startswith('MACCS_')])

    from rdkit.Chem import MACCSkeys
    fp   = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=mfp_size)
    mfp  = {f'MFP_{i}': int(fp[i]) for i in range(mfp_size)}
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs = {f'MACCS_{i}': int(maccs_fp[i]) for i in range(maccs_size)}

    all_feats = {**basic, **mfp, **maccs, **tox_flags}
    row = pd.DataFrame([all_feats]).reindex(columns=feat_cols, fill_value=0)

    predictions = {}
    for target, model_data in models.items():
        prob = model_data['model'].predict_proba(row)[0][1]
        predictions[target] = round(prob, 3)

    return predictions

real_drugs = {
    'Aspirin':     'CC(=O)Oc1ccccc1C(=O)O',
    'Ibuprofen':   'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'Estradiol':   'OC1=CC2=C(CCC3C2CCC4(C)C3CCC4O)C=C1',
    'Testosterone':'CC12CCC3c4ccc(O)cc4CCC3C1CCC2=O',
    'Caffeine':    'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'Metformin':   'CN(C)C(=N)NC(=N)N',
}

drug_results = {}
for drug_name, smiles in real_drugs.items():
    preds = predict_molecule(smiles, models)
    if preds:
        drug_results[drug_name] = preds
        print(f"  ✅ Predicted: {drug_name}")

drug_df = pd.DataFrame(drug_results).T
drug_df.to_csv('results/real_drug_predictions.csv')

plt.figure(figsize=(14, 6))
sns.heatmap(
    drug_df,
    annot=True,
    fmt=".2f",
    cmap='RdYlGn_r',
    vmin=0, vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Toxicity Probability"}
)
plt.title("Toxicity Risk Across Real FDA-Approved Drugs", fontsize=13)
plt.tight_layout()
plt.savefig('results/05_real_drug_predictions.png', dpi=150)
plt.close()
print("  ✅ Saved 05_real_drug_predictions.png")

# ----------------------------------------
# PLOT 6: ROC Curves Per Assay
# ----------------------------------------
print("\n[6/6] Generating ROC Curves for all 12 assays...")

feat_cols_main = models[list(models.keys())[0]]['feature_cols']

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()

for i, target in enumerate(tox_targets):
    if target not in df.columns:
        continue

    df_t = df[feat_cols_main + [target]].dropna()
    X    = df_t[feat_cols_main]
    y    = df_t[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_proba = models[target]['model'].predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc_score(fpr, tpr)

    axes[i].plot(fpr, tpr, color='#e74c3c', lw=2.5,
                 label=f'AUC = {roc_auc:.3f}')
    axes[i].plot([0, 1], [0, 1], color='gray',
                 linestyle='--', lw=1, alpha=0.5)
    axes[i].fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')

    color = '#2ecc71' if roc_auc >= 0.75 else '#e74c3c'
    axes[i].set_title(target, fontsize=11, color=color, fontweight='bold')
    axes[i].set_xlabel('False Positive Rate', fontsize=8)
    axes[i].set_ylabel('True Positive Rate', fontsize=8)
    axes[i].legend(fontsize=9, loc='lower right')
    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([0, 1.02])
    axes[i].grid(True, alpha=0.2)

plt.suptitle('ROC Curves — All 12 Tox21 Assays\n(ToxPredict Ensemble Model)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/06_roc_curves.png', dpi=150)
plt.close()
print("  ✅ Saved 06_roc_curves.png")

# ----------------------------------------
# FINAL SUMMARY
# ----------------------------------------
print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 60)
print("\nFiles saved in results/:")
for f in sorted(os.listdir('results')):
    size = os.path.getsize(f'results/{f}') // 1024
    print(f"  ✅ {f:45s} ({size} KB)")

print(f"\n📊 Model Summary:")
print(f"   Best AUC   : {results_df['ROC-AUC'].max():.4f} ({results_df['ROC-AUC'].idxmax()})")
print(f"   Mean AUC   : {results_df['ROC-AUC'].mean():.4f}")
print(f"   Above 0.75 : {(results_df['ROC-AUC'] > 0.75).sum()}/12 targets")

print("\n🎉 Visualization Complete!")