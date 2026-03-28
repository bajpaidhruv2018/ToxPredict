# ============================================
# STREAMLIT APP — TOXPREDICT
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import urllib.parse
import urllib.request
import json

sys.path.append('..')

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem, MolFromSmarts

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(
    page_title="ToxPredict",
    page_icon="🧬",
    layout="wide"
)

# ----------------------------------------
# Load models
# ----------------------------------------
@st.cache_resource
def load_models():
    with open('results/models.pkl', 'rb') as f:
        return pickle.load(f)

models = load_models()

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

# ----------------------------------------
# Helper functions
# ----------------------------------------
def get_features(smiles, feat_cols):
    try:
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

        fp      = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_dict = {f'FP_{i}': int(fp[i]) for i in range(1024)}

        all_feats = {**basic, **fp_dict, **tox_flags}
        row       = pd.DataFrame([all_feats])
        row       = row.reindex(columns=feat_cols, fill_value=0)
        return row, mol, basic, tox_flags

    except:
        return None


def lookup_pubchem(drug_name):
    """
    Tries 3 different APIs in order until one works
    """
    import requests
    name = drug_name.strip()

    # --- API 1: PubChem ---
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200:
            data = r.json()
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass

    # --- API 2: ChemSpider via OPSIN (no key needed) ---
    try:
        url = f"https://opsin.ch.cam.ac.uk/opsin/{urllib.parse.quote(name)}.json"
        r = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200:
            data = r.json()
            if 'smiles' in data:
                return data['smiles']
    except:
        pass

    # --- API 3: CIR (Chemical Identifier Resolver) ---
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(name)}/smiles"
        r = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except:
        pass

    return None


# ----------------------------------------
# Initialize session state
# ----------------------------------------
if 'current_smiles' not in st.session_state:
    st.session_state['current_smiles'] = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin default

# ----------------------------------------
# Header
# ----------------------------------------
st.title("🧬 ToxPredict")
st.markdown("### AI-Powered Drug Toxicity Predictor")
st.markdown("Predict toxicity of **any molecule** across 12 biological targets instantly.")
st.markdown("---")

# ----------------------------------------
# Preset drug buttons
# ----------------------------------------
st.markdown("**Quick test with real drugs:**")
c1, c2, c3, c4, c5, c6 = st.columns(6)

presets = {
    "💊 Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "☕ Caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "🤕 Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "💉 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "💙 Metformin":   "CN(C)C(=N)NC(=N)N",
    "🩸 Warfarin":    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}

for col, (label, smi) in zip([c1, c2, c3, c4, c5, c6], presets.items()):
    if col.button(label):
        st.session_state['current_smiles'] = smi

st.markdown("---")

# ----------------------------------------
# Input tabs
# ----------------------------------------
tab1, tab2 = st.tabs(["🔤 Search by Drug Name", "🧪 Enter SMILES Manually"])

with tab1:
    st.markdown("Type any drug name and we'll fetch its structure from PubChem automatically.")
    drug_name_input = st.text_input(
        "Drug name:",
        placeholder="e.g. Metformin, Penicillin, Warfarin, Thalidomide, Fentanyl..."
    )

    if st.button("🔍 Find Drug", key="search_btn"):
        if drug_name_input.strip() == "":
            st.warning("Please enter a drug name.")
        else:
            with st.spinner(f"Looking up '{drug_name_input}' on PubChem..."):
                result_smiles = lookup_pubchem(drug_name_input)

            if result_smiles:
                st.session_state['current_smiles'] = result_smiles
                st.success(f"✅ Found **{drug_name_input}**!")
                st.code(result_smiles, language="text")
            else:
                st.error(f"❌ Could not find **'{drug_name_input}'**. Try a different spelling or use the SMILES tab.")

with tab2:
    st.markdown("Paste a SMILES string directly if you have one.")
    manual_smiles = st.text_input(
        "SMILES String:",
        value=st.session_state['current_smiles'],
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
    )
    if st.button("Load this SMILES", key="load_smiles_btn"):
        st.session_state['current_smiles'] = manual_smiles

# Show currently loaded molecule
st.markdown("---")
st.info(f"**Currently loaded molecule:** `{st.session_state['current_smiles']}`")

# ----------------------------------------
# Predict button
# ----------------------------------------
if st.button("🔬 Predict Toxicity", type="primary"):

    smiles_input = st.session_state['current_smiles']
    feat_cols    = models[list(models.keys())[0]]['feature_cols']
    result       = get_features(smiles_input, feat_cols)

    if result is None:
        st.error("❌ Could not process this molecule. Please check the SMILES string.")

    else:
        row, mol, basic_props, tox_flags = result

        col1, col2 = st.columns([1, 2])

        # ---- Left column: structure + properties ----
        with col1:
            st.markdown("**Molecule Structure**")
            img = Draw.MolToImage(mol, size=(280, 220))
            st.image(img)

            st.markdown("**Molecular Properties**")
            props_df = pd.DataFrame(
                basic_props.items(),
                columns=['Property', 'Value']
            )
            props_df['Value'] = props_df['Value'].round(3)
            st.dataframe(props_df, hide_index=True, use_container_width=True)

        # ---- Right column: predictions ----
        with col2:
            st.markdown("**Toxicity Predictions Across 12 Assays**")

            predictions = {}
            for target, model_data in models.items():
                prob = model_data['model'].predict_proba(row)[0][1]
                predictions[target] = prob

            avg_risk = np.mean(list(predictions.values()))

            # Overall risk banner
            if avg_risk > 0.5:
                st.error(f"⚠️ Overall Risk Score: {avg_risk:.1%} — HIGH RISK")
            elif avg_risk > 0.3:
                st.warning(f"⚠️ Overall Risk Score: {avg_risk:.1%} — MODERATE RISK")
            else:
                st.success(f"✅ Overall Risk Score: {avg_risk:.1%} — LOW RISK")

            # Per target metric cards
            pred_cols = st.columns(3)
            for i, (target, prob) in enumerate(predictions.items()):
                with pred_cols[i % 3]:
                    icon = "🔴" if prob > 0.5 else "🟢"
                    st.metric(
                        label=f"{icon} {target}",
                        value=f"{prob:.1%}"
                    )

        # ---- Toxicophore alerts ----
        st.markdown("---")
        st.markdown("**🔍 Toxicophore Analysis**")

        found = {
            k.replace('tox_', '').replace('_', ' '): v
            for k, v in tox_flags.items()
            if v == 1 and k != 'total_toxicophores'
        }

        if found:
            st.warning(f"⚠️ Found **{len(found)} toxicophore(s)** in this molecule:")
            for name in found:
                st.markdown(f"  - 🔴 **{name}** — known toxic substructure")
        else:
            st.success("✅ No known toxicophores detected in this molecule.")

        # ---- ADMET Profile ----
        st.markdown("---")
        st.markdown("**💊 ADMET Drug Safety Profile**")

        mw   = basic_props['MolWt']
        logp = basic_props['LogP']
        hbd  = basic_props['NumHDonors']
        hba  = basic_props['NumHAcceptors']
        tpsa = basic_props['TPSA']

        admet = {
            'Oral Bioavailability':
                '✅ Good'    if mw < 500 and logp < 5 and hbd <= 5 and hba <= 10
                else '⚠️ Poor',
            'GI Absorption':
                '✅ High'    if tpsa < 140 else '⚠️ Low',
            'BBB Penetration':
                '✅ Likely'  if logp > 0 and tpsa < 90 and mw < 450
                else '⚠️ Unlikely',
            'hERG Cardiac Risk':
                '⚠️ High'   if logp > 3.7 and mw > 400 else '✅ Low',
            'Hepatotoxicity Risk':
                '⚠️ High'   if logp > 4 else '✅ Low',
            'Lipinski Compliant':
                '✅ Yes'     if mw<=500 and logp<=5 and hbd<=5 and hba<=10
                else '⚠️ No',
            'Lead-Like':
                '✅ Yes'     if mw<=350 and logp<=3 and hbd<=3 and hba<=7
                else '⚠️ No',
        }

        a1, a2 = st.columns(2)
        items  = list(admet.items())

        with a1:
            for k, v in items[:4]:
                st.markdown(f"**{k}:** {v}")
        with a2:
            for k, v in items[4:]:
                st.markdown(f"**{k}:** {v}")

        # ---- Download report ----
        st.markdown("---")
        report = pd.DataFrame({
            'Target': list(predictions.keys()),
            'Toxicity Probability': [f"{v:.1%}" for v in predictions.values()],
            'Risk Level': ['HIGH' if v > 0.5 else 'LOW' for v in predictions.values()]
        })
        csv = report.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Report as CSV",
            data=csv,
            file_name="toxicity_report.csv",
            mime="text/csv"
        )