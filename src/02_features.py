# ============================================
# STEP 2: EXTRACT MOLECULAR FEATURES
# ============================================
# Refactored to process scaffold-split files
# independently (train and test are kept separate
# to prevent any form of data leakage).
# ============================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MolFromSmarts
import os

os.makedirs('results', exist_ok=True)

# ────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTIONS
# ────────────────────────────────────────────

def get_basic_features(smiles):
    """Compute 9 basic molecular descriptors."""
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


def get_morgan_fp(smiles):
    """Compute 1024-bit Morgan (ECFP4) fingerprint."""
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
    """Detect 10 known toxicophore substructures."""
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


# ────────────────────────────────────────────
# PROCESS A SINGLE SPLIT
# ────────────────────────────────────────────

def process_split(input_path: str, output_path: str, split_name: str):
    """
    Load a scaffold-split CSV, compute all features,
    and save the processed result.
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print(f"{'=' * 60}")

    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} molecules from {input_path}")

    # --- Basic descriptors ---
    print(f"\n  Extracting basic descriptors...")
    basic_list = []
    for i, smiles in enumerate(df['smiles']):
        result = get_basic_features(smiles)
        if result is None:
            result = {
                'MolWt': 0, 'LogP': 0, 'NumHDonors': 0,
                'NumHAcceptors': 0, 'TPSA': 0, 'NumRotatableBonds': 0,
                'NumAromaticRings': 0, 'NumHeavyAtoms': 0, 'FractionCSP3': 0
            }
        basic_list.append(result)
        if i % 2000 == 0:
            print(f"    Progress: {i}/{len(df)}")
    basic_df = pd.DataFrame(basic_list)
    print(f"  ✅ Basic descriptors — {basic_df.shape[1]} features")

    # --- Morgan fingerprints ---
    print(f"\n  Extracting Morgan fingerprints...")
    fp_list = []
    for i, smiles in enumerate(df['smiles']):
        fp = get_morgan_fp(smiles)
        if fp is None:
            fp = np.zeros(1024)
        fp_list.append(fp)
        if i % 2000 == 0:
            print(f"    Progress: {i}/{len(df)}")
    fp_df = pd.DataFrame(fp_list, columns=[f'FP_{i}' for i in range(1024)])
    print(f"  ✅ Morgan fingerprints — {fp_df.shape[1]} features")

    # --- Toxicophores ---
    print(f"\n  Detecting toxicophores...")
    empty_tox = {name: 0 for name in TOXICOPHORES.keys()}
    empty_tox['total_toxicophores'] = 0
    tox_list = []
    for i, smiles in enumerate(df['smiles']):
        result = detect_toxicophores(smiles)
        if result is None:
            result = empty_tox.copy()
        tox_list.append(result)
        if i % 2000 == 0:
            print(f"    Progress: {i}/{len(df)}")
    tox_df = pd.DataFrame(tox_list)
    print(f"  ✅ Toxicophores — {tox_df.shape[1]} features")

    # --- Combine and save ---
    final_df = pd.concat([df, basic_df, fp_df, tox_df], axis=1)
    final_df = final_df.dropna(subset=['MolWt'])

    final_df.to_csv(output_path, index=False)
    total_feats = basic_df.shape[1] + fp_df.shape[1] + tox_df.shape[1]
    print(f"\n  ✅ Total features: {total_feats}")
    print(f"  ✅ Final shape: {final_df.shape}")
    print(f"  ✅ Saved to {output_path}")

    return final_df


# ────────────────────────────────────────────
# MAIN: Process both train and test splits
# ────────────────────────────────────────────

print("=" * 60)
print("STEP 2: FEATURE EXTRACTION (SCAFFOLD-SPLIT)")
print("=" * 60)

train_df = process_split(
    input_path  = 'data/tox21_train_scaffold.csv',
    output_path = 'data/tox21_train_processed.csv',
    split_name  = 'train'
)

test_df = process_split(
    input_path  = 'data/tox21_test_scaffold.csv',
    output_path = 'data/tox21_test_processed.csv',
    split_name  = 'test'
)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Train processed : {train_df.shape}")
print(f"  Test  processed : {test_df.shape}")
print("\n🎉 Feature Extraction Complete!")