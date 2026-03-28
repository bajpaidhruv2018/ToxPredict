# ============================================
# STEP 2: EXTRACT MOLECULAR FEATURES
# ============================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MolFromSmarts
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)
df = pd.read_csv('data/tox21.csv')
print(f"✅ Loaded {len(df)} molecules")

# ----------------------------------------
# FEATURE 1: Basic Molecular Descriptors
# ----------------------------------------
print("\n" + "=" * 50)
print("EXTRACTING BASIC DESCRIPTORS")
print("=" * 50)

def get_basic_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
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
    except:
        return None

basic_list = []
for i, smiles in enumerate(df['smiles']):
    result = get_basic_features(smiles)
    # If molecule is invalid, fill with zeros instead of None
    if result is None:
        result = {
            'MolWt': 0, 'LogP': 0, 'NumHDonors': 0,
            'NumHAcceptors': 0, 'TPSA': 0, 'NumRotatableBonds': 0,
            'NumAromaticRings': 0, 'NumHeavyAtoms': 0, 'FractionCSP3': 0
        }
    basic_list.append(result)
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

basic_df = pd.DataFrame(basic_list)
print(f"✅ Basic descriptors done — {basic_df.shape[1]} features")

# ----------------------------------------
# FEATURE 2: Morgan Fingerprints
# ----------------------------------------
print("\n" + "=" * 50)
print("EXTRACTING MORGAN FINGERPRINTS")
print("=" * 50)

def get_morgan_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=1024
        )
        return np.array(fp)
    except:
        return None

fp_list = []
for i, smiles in enumerate(df['smiles']):
    fp = get_morgan_fp(smiles)
    # If invalid molecule, use zeros
    if fp is None:
        fp = np.zeros(1024)
    fp_list.append(fp)
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

fp_df = pd.DataFrame(
    fp_list,
    columns=[f'FP_{i}' for i in range(1024)]
)
print(f"✅ Morgan fingerprints done — {fp_df.shape[1]} features")

# ----------------------------------------
# FEATURE 3: Toxicophore Detection
# ----------------------------------------
print("\n" + "=" * 50)
print("DETECTING TOXICOPHORES")
print("=" * 50)

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

def detect_toxicophores(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        results = {}
        for name, smarts in TOXICOPHORES.items():
            pattern = MolFromSmarts(smarts)
            if pattern:
                results[name] = int(mol.HasSubstructMatch(pattern))
        results['total_toxicophores'] = sum(results.values())
        return results
    except:
        return None

empty_tox = {name: 0 for name in TOXICOPHORES.keys()}
empty_tox['total_toxicophores'] = 0

tox_list = []
for i, smiles in enumerate(df['smiles']):
    result = detect_toxicophores(smiles)
    if result is None:
        result = empty_tox.copy()
    tox_list.append(result)
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

tox_df = pd.DataFrame(tox_list)
print(f"✅ Toxicophores done — {tox_df.shape[1]} features")

print("\nToxicophore counts in dataset:")
for col in tox_df.columns:
    print(f"  {col:30s}: {int(tox_df[col].sum())}")

# ----------------------------------------
# COMBINE ALL FEATURES
# ----------------------------------------
print("\n" + "=" * 50)
print("COMBINING ALL FEATURES")
print("=" * 50)

final_df = pd.concat([df, basic_df, fp_df, tox_df], axis=1)
final_df = final_df.dropna(subset=['MolWt'])

final_df.to_csv('data/tox21_processed.csv', index=False)

print(f"✅ Total features: {basic_df.shape[1] + fp_df.shape[1] + tox_df.shape[1]}")
print(f"✅ Final dataset shape: {final_df.shape}")
print(f"✅ Saved to data/tox21_processed.csv")
print("\n🎉 Feature Extraction Complete!")