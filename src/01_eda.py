# ============================================
# STEP 1: EXPLORE THE DATA
# ============================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results folder if not exists
os.makedirs('results', exist_ok=True)

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv('data/tox21.csv')

print(f"✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

print("\n" + "=" * 50)
print("FIRST 3 ROWS")
print("=" * 50)
print(df.head(3))

print("\n" + "=" * 50)
print("MISSING VALUES PER COLUMN")
print("=" * 50)
print(df.isnull().sum())

# Define the 12 toxicity targets
tox_targets = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Only keep targets that exist in our dataset
tox_targets = [t for t in tox_targets if t in df.columns]

print("\n" + "=" * 50)
print("CLASS BALANCE PER TOXICITY TARGET")
print("=" * 50)

for target in tox_targets:
    counts  = df[target].value_counts()
    missing = df[target].isnull().sum()
    toxic   = counts.get(1.0, 0)
    nontox  = counts.get(0.0, 0)
    print(f"  {target:20s} | Toxic={toxic:4d} | Non-toxic={nontox:4d} | Missing={missing}")

print("\n" + "=" * 50)
print("SAVING CLASS BALANCE CHART")
print("=" * 50)

# Plot class balance
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, target in enumerate(tox_targets):
    counts = df[target].value_counts()
    axes[i].bar(
        ['Non-toxic', 'Toxic'],
        [counts.get(0.0, 0), counts.get(1.0, 0)],
        color=['#2ecc71', '#e74c3c']
    )
    axes[i].set_title(target)
    axes[i].set_ylabel('Count')

plt.suptitle('Class Balance Across 12 Tox21 Assays', fontsize=14)
plt.tight_layout()
plt.savefig('results/class_balance.png', dpi=150)
print("✅ Saved results/class_balance.png")

print("\n🎉 EDA Complete!")