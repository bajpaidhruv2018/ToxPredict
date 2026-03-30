# ============================================
# STEP 2: EXTRACT MOLECULAR FEATURES
# Updated: Morgan 2048-bit + MACCS Keys + Toxicophores
# ============================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MolFromSmarts, MACCSkeys
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
df = pd.read_csv('data/tox21.csv')
print(f"✅ Loaded {len(df)} molecules")

# ----------------------------------------
# FEATURE 1: Basic Molecular Descriptors
# ----------------------------------------
print("\n" + "=" * 60)
print("EXTRACTING BASIC DESCRIPTORS (9 features)")
print("=" * 60)

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

empty_basic = {
    'MolWt': 0, 'LogP': 0, 'NumHDonors': 0,
    'NumHAcceptors': 0, 'TPSA': 0, 'NumRotatableBonds': 0,
    'NumAromaticRings': 0, 'NumHeavyAtoms': 0, 'FractionCSP3': 0
}

basic_list = []
for i, smiles in enumerate(df['smiles']):
    result = get_basic_features(smiles)
    if result is None:
        result = empty_basic.copy()
    basic_list.append(result)
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

basic_df = pd.DataFrame(basic_list)
print(f"✅ Basic descriptors done — {basic_df.shape[1]} features")

# ----------------------------------------
# FEATURE 2: Morgan Fingerprints (2048-bit)
# ----------------------------------------
print("\n" + "=" * 60)
print("EXTRACTING MORGAN FINGERPRINTS 2048-bit (upgraded from 1024)")
print("=" * 60)

def get_morgan_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Upgraded to 2048 bits for better molecular representation
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048
        )
        return np.array(fp)
    except:
        return None

fp_list = []
for i, smiles in enumerate(df['smiles']):
    fp = get_morgan_fp(smiles)
    if fp is None:
        fp = np.zeros(2048)
    fp_list.append(fp)
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

fp_df = pd.DataFrame(
    fp_list,
    columns=[f'MFP_{i}' for i in range(2048)]
)
print(f"✅ Morgan fingerprints done — {fp_df.shape[1]} features")

# ----------------------------------------
# FEATURE 3: MACCS Keys (166 features)
# ----------------------------------------
print("\n" + "=" * 60)
print("EXTRACTING MACCS KEYS (166 chemist-designed features)")
print("=" * 60)

def get_maccs_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except:
        return np.zeros(167)

maccs_list = []
for i, smiles in enumerate(df['smiles']):
    maccs_list.append(get_maccs_fp(smiles))
    if i % 2000 == 0:
        print(f"  Progress: {i}/{len(df)}")

maccs_df = pd.DataFrame(
    maccs_list,
    columns=[f'MACCS_{i}' for i in range(167)]
)
print(f"✅ MACCS keys done — {maccs_df.shape[1]} features")

# ----------------------------------------
# FEATURE 4: Toxicophore Detection
# ----------------------------------------
print("\n" + "=" * 60)
print("DETECTING TOXICOPHORES (10 toxic substructures)")
print("=" * 60)

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

print("\nToxicophore frequency in dataset:")
for col in tox_df.columns:
    if col != 'total_toxicophores':
        count = int(tox_df[col].sum())
        pct   = count / len(tox_df) * 100
        print(f"  {col:30s}: {count:4d} molecules ({pct:.1f}%)")

# ----------------------------------------
# COMBINE ALL FEATURES
# ----------------------------------------
print("\n" + "=" * 60)
print("COMBINING ALL FEATURES")
print("=" * 60)

final_df = pd.concat([df, basic_df, fp_df, maccs_df, tox_df], axis=1)
final_df = final_df.dropna(subset=['MolWt'])

final_df.to_csv('data/tox21_processed.csv', index=False)

total_features = basic_df.shape[1] + fp_df.shape[1] + maccs_df.shape[1] + tox_df.shape[1]
print(f"\n✅ Basic descriptors : {basic_df.shape[1]}")
print(f"✅ Morgan FP (2048)  : {fp_df.shape[1]}")
print(f"✅ MACCS Keys        : {maccs_df.shape[1]}")
print(f"✅ Toxicophores      : {tox_df.shape[1]}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"✅ TOTAL FEATURES    : {total_features}")
print(f"✅ Final dataset     : {final_df.shape}")
print(f"✅ Saved to data/tox21_processed.csv")
print("\n🎉 Feature Extraction Complete!")