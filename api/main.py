# ============================================
# TOXPREDICT BACKEND API
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request
import json
import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors,
    AllChem, MolFromSmarts, MACCSkeys
)

# ----------------------------------------
# App setup
# ----------------------------------------
app = FastAPI(title="ToxPredict API", version="1.0.0")

# Allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load models once at startup
# ----------------------------------------
print("Loading models...")
with open("results/models.pkl", "rb") as f:
    MODELS = pickle.load(f)
print(f"✅ Loaded {len(MODELS)} models")

# ----------------------------------------
# Constants
# ----------------------------------------
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

TOXICOPHORE_DESCRIPTIONS = {
    'tox_Nitro_group':      'DNA damage and mutagenicity',
    'tox_Aldehyde':         'Protein binding and cell damage',
    'tox_Epoxide':          'DNA alkylation and carcinogenicity',
    'tox_Aromatic_amine':   'Metabolic activation and carcinogenicity',
    'tox_Hydrazine':        'Hepatotoxicity and carcinogenicity',
    'tox_Alkyl_halide':     'Reactive electrophile',
    'tox_Michael_acceptor': 'Covalent protein binding',
    'tox_Quinone':          'Oxidative stress and redox cycling',
    'tox_Azo_compound':     'Metabolic activation to amines',
    'tox_Peroxide':         'Oxidative DNA damage',
}

DRUG_DATABASE = {
    "aspirin":          "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":      "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen":    "CC(=O)Nc1ccc(O)cc1",
    "caffeine":         "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "metformin":        "CN(C)C(=N)NC(=N)N",
    "warfarin":         "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "morphine":         "CN1CCC23c4c(ccc(O)c4OC2C(O)C=CC3)C1",
    "heroin":           "CC(=O)Oc1ccc2CC3N(C)CCC3=CC2c1OC(C)=O",
    "cocaine":          "COC(=O)C1CC2CCC1N2C",
    "nicotine":         "CN1CCCC1c1cccnc1",
    "ethanol":          "CCO",
    "thalidomide":      "O=C1CCC(=O)N1C1CCC(=O)N(C1=O)c1ccccc1",
    "testosterone":     "CC12CCC3c4ccc(O)cc4CCC3C1CCC2=O",
    "estradiol":        "OC1=CC2=C(CCC3C2CCC4(C)C3CCC4O)C=C1",
    "caffeine":         "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "doxorubicin":      "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)CC4c(O)c3C(=O)c12",
    "cyclophosphamide": "ClCCN(CCCl)P1(=O)NCCCO1",
    "nitrobenzene":     "O=[N+]([O-])c1ccccc1",
    "aniline":          "Nc1ccccc1",
    "benzene":          "c1ccccc1",
    "fentanyl":         "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1",
    "methotrexate":     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O",
    "tamoxifen":        "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "ciprofloxacin":    "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "fluconazole":      "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
    "diazepam":         "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "viagra":           "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "remdesivir":       "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(O)C(C(N2C=CC(=O)NC2=O)O1)O)Oc1ccccc1",
}

# ----------------------------------------
# Input/Output Models
# ----------------------------------------
class DrugNameRequest(BaseModel):
    drug_name: str

class SmilesRequest(BaseModel):
    smiles: str

# ----------------------------------------
# Helper Functions
# ----------------------------------------
def lookup_smiles(drug_name: str):
    """Convert drug name to SMILES"""
    key = drug_name.strip().lower()

    # Check local database first
    if key in DRUG_DATABASE:
        return DRUG_DATABASE[key]

    # Try PubChem
    try:
        encoded = urllib.parse.quote(drug_name.strip())
        url  = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON"
        req  = urllib.request.urlopen(url, timeout=8)
        data = json.loads(req.read().decode())
        return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass

    # Try CIR
    try:
        encoded = urllib.parse.quote(drug_name.strip())
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles"
        req = urllib.request.urlopen(url, timeout=8)
        result = req.read().decode().strip()
        if result:
            return result
    except:
        pass

    return None


def extract_features(smiles: str):
    """Extract all molecular features from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    # Basic descriptors
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

    # Toxicophores
    tox_flags = {}
    for name, smarts in TOXICOPHORES.items():
        pattern = MolFromSmarts(smarts)
        tox_flags[name] = int(
            mol.HasSubstructMatch(pattern)
        ) if pattern else 0
    tox_flags['total_toxicophores'] = sum(tox_flags.values())

    # Get feature columns from first model
    feat_cols = MODELS[list(MODELS.keys())[0]]['feature_cols']
    mfp_size  = len([c for c in feat_cols if c.startswith('MFP_')])
    maccs_size = len([c for c in feat_cols if c.startswith('MACCS_')])

    # Morgan fingerprints
    fp      = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=max(mfp_size, 1024)
    )
    fp_dict = {f'MFP_{i}': int(fp[i]) for i in range(mfp_size)}

    # MACCS keys
    maccs_fp   = MACCSkeys.GenMACCSKeys(mol)
    maccs_dict = {
        f'MACCS_{i}': int(maccs_fp[i])
        for i in range(min(maccs_size, 167))
    }

    # Combine all
    all_feats = {**basic, **fp_dict, **maccs_dict, **tox_flags}
    row = pd.DataFrame([all_feats]).reindex(
        columns=feat_cols, fill_value=0
    )

    return row, mol, basic, tox_flags


def run_prediction(smiles: str):
    """Run full prediction pipeline"""
    row, mol, basic, tox_flags = extract_features(smiles)

    if row is None:
        return None

    # Get predictions from all 12 models
    predictions = {}
    for target, model_data in MODELS.items():
        prob = model_data['model'].predict_proba(row)[0][1]
        predictions[target] = round(float(prob), 4)

    avg_risk = float(np.mean(list(predictions.values())))

    # Risk level
    if avg_risk > 0.5:
        risk_level = "HIGH"
        risk_color = "#ef4444"
    elif avg_risk > 0.3:
        risk_level = "MODERATE"
        risk_color = "#f59e0b"
    else:
        risk_level = "LOW"
        risk_color = "#10b981"

    # Toxicophores found
    found_tox = {
        k.replace('tox_', '').replace('_', ' '): 
        TOXICOPHORE_DESCRIPTIONS.get(k, '')
        for k, v in tox_flags.items()
        if v == 1 and k != 'total_toxicophores'
    }

    # ADMET profile
    mw   = basic['MolWt']
    logp = basic['LogP']
    hbd  = basic['NumHDonors']
    hba  = basic['NumHAcceptors']
    tpsa = basic['TPSA']

    admet = {
        'oral_bioavailability':
            'Good' if mw<500 and logp<5 and hbd<=5 and hba<=10
            else 'Poor',
        'gi_absorption':
            'High' if tpsa < 140 else 'Low',
        'bbb_penetration':
            'Likely' if logp>0 and tpsa<90 and mw<450
            else 'Unlikely',
        'herg_cardiac_risk':
            'High' if logp>3.7 and mw>400 else 'Low',
        'hepatotoxicity_risk':
            'High' if logp > 4 else 'Low',
        'lipinski_compliant':
            'Yes' if mw<=500 and logp<=5 and hbd<=5 and hba<=10
            else 'No',
    }

    return {
        "smiles":          smiles,
        "predictions":     predictions,
        "overall_risk":    round(avg_risk, 4),
        "risk_level":      risk_level,
        "risk_color":      risk_color,
        "molecular_props": {k: round(v, 3) for k, v in basic.items()},
        "toxicophores":    found_tox,
        "admet":           admet,
    }


# ----------------------------------------
# API ENDPOINTS
# ----------------------------------------

@app.get("/")
def root():
    return {
        "name":    "ToxPredict API",
        "version": "1.0.0",
        "status":  "running",
        "models":  len(MODELS),
        "targets": list(MODELS.keys())
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(MODELS)}

@app.post("/lookup")
def lookup_drug(request: DrugNameRequest):
    """Convert drug name to SMILES"""
    smiles = lookup_smiles(request.drug_name)
    if smiles:
        return {"success": True, "smiles": smiles}
    return {"success": False, "error": f"Could not find '{request.drug_name}'"}

@app.post("/predict")
def predict(request: SmilesRequest):
    """Predict toxicity from SMILES string"""
    try:
        result = run_prediction(request.smiles)
        if result is None:
            return {
                "success": False,
                "error": "Invalid SMILES string"
            }
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict-by-name")
def predict_by_name(request: DrugNameRequest):
    """Lookup drug name then predict — one step"""
    smiles = lookup_smiles(request.drug_name)
    if not smiles:
        return {
            "success": False,
            "error": f"Could not find '{request.drug_name}'"
        }
    try:
        result = run_prediction(smiles)
        if result is None:
            return {"success": False, "error": "Invalid molecule"}
        return {
            "success":   True,
            "drug_name": request.drug_name,
            **result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}