# ============================================
# STEP 1b: SCAFFOLD-BASED TRAIN/TEST SPLIT
# ============================================
# This script fixes data leakage by ensuring that
# molecules sharing the same Murcko scaffold are
# forced into the SAME split (train OR test, never both).
# ============================================

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupShuffleSplit
import os

os.makedirs('data', exist_ok=True)

# ────────────────────────────────────────────
# 1) Load raw dataset
# ────────────────────────────────────────────
print("=" * 60)
print("LOADING RAW DATASET")
print("=" * 60)

df = pd.read_csv('data/tox21.csv')
print(f"✅ Loaded {len(df)} molecules  |  Columns: {df.shape[1]}")


# ────────────────────────────────────────────
# 2) Compute Murcko Scaffold for each molecule
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPUTING MURCKO SCAFFOLDS")
print("=" * 60)

def get_murcko_scaffold(smiles: str) -> str:
    """
    Return the generic Murcko scaffold for a SMILES string.
    Returns 'Invalid' if the SMILES cannot be parsed by RDKit.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid"
        # includeChirality=False for a more general scaffold
        scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        return scaffold
    except Exception:
        return "Invalid"


df['scaffold'] = df['smiles'].apply(get_murcko_scaffold)

# Report invalid SMILES
n_invalid = (df['scaffold'] == 'Invalid').sum()
print(f"   Total molecules     : {len(df)}")
print(f"   Invalid SMILES      : {n_invalid}")

# Drop invalid rows
df = df[df['scaffold'] != 'Invalid'].reset_index(drop=True)
print(f"   After filtering     : {len(df)} molecules retained")
print(f"   Unique scaffolds    : {df['scaffold'].nunique()}")


# ────────────────────────────────────────────
# 3) Scaffold-aware GroupShuffleSplit
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("PERFORMING SCAFFOLD SPLIT (80/20)")
print("=" * 60)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

# Groups = scaffold → all molecules with same scaffold go together
train_idx, test_idx = next(splitter.split(X=df, y=None, groups=df['scaffold']))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

print(f"   Train set : {len(train_df)} molecules")
print(f"   Test  set : {len(test_df)} molecules")
print(f"   Split     : {len(train_df)/len(df)*100:.1f}% / {len(test_df)/len(df)*100:.1f}%")


# ────────────────────────────────────────────
# 4) Validate: zero scaffold leakage
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("LEAKAGE VALIDATION")
print("=" * 60)

train_scaffolds = set(train_df['scaffold'].unique())
test_scaffolds  = set(test_df['scaffold'].unique())
overlap         = train_scaffolds & test_scaffolds

print(f"   Unique scaffolds in train : {len(train_scaffolds)}")
print(f"   Unique scaffolds in test  : {len(test_scaffolds)}")
print(f"   Scaffold overlap          : {len(overlap)}")

if len(overlap) == 0:
    print("   ✅ PASS — Zero scaffold leakage between train and test!")
else:
    print("   ❌ FAIL — Scaffold leakage detected! Overlapping scaffolds:")
    for s in list(overlap)[:10]:
        print(f"      {s}")


# ────────────────────────────────────────────
# 5) Save the splits
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING SPLIT FILES")
print("=" * 60)

train_df.to_csv('data/tox21_train_scaffold.csv', index=False)
test_df.to_csv('data/tox21_test_scaffold.csv',  index=False)

print(f"   ✅ data/tox21_train_scaffold.csv  ({len(train_df)} rows)")
print(f"   ✅ data/tox21_test_scaffold.csv   ({len(test_df)} rows)")

print("\n🎉 Scaffold Split Complete!")
