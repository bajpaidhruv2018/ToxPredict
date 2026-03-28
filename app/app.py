# ============================================
# TOXPREDICT — AI Drug Toxicity Predictor
# Complete Updated Version
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
import requests

sys.path.append('..')

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Draw, Descriptors, rdMolDescriptors,
    AllChem, MolFromSmarts, MolToMolBlock
)

import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="ToxPredict",
    page_icon="🧬",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
    }
    .risk-high {
        background: #3d1515;
        border: 1px solid #e74c3c;
        border-radius: 10px;
        padding: 15px;
        color: #e74c3c;
        font-weight: bold;
        font-size: 18px;
    }
    .risk-moderate {
        background: #3d2e10;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 15px;
        color: #f39c12;
        font-weight: bold;
        font-size: 18px;
    }
    .risk-low {
        background: #0f3d20;
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 15px;
        color: #2ecc71;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTS
# ============================================
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

DRUG_DATABASE = {
    "aspirin":          "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":      "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen":    "CC(=O)Nc1ccc(O)cc1",
    "caffeine":         "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "metformin":        "CN(C)C(=N)NC(=N)N",
    "warfarin":         "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "penicillin":       "CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1",
    "amoxicillin":      "CC1(C)SC2C(NC1=O)C(=O)N2C(C(=O)O)c1ccc(N)cc1",
    "morphine":         "CN1CCC23c4c(ccc(O)c4OC2C(O)C=CC3)C1",
    "heroin":           "CC(=O)Oc1ccc2CC3N(C)CCC3=CC2c1OC(C)=O",
    "cocaine":          "COC(=O)C1CC2CCC1N2C",
    "nicotine":         "CN1CCCC1c1cccnc1",
    "ethanol":          "CCO",
    "glucose":          "OCC1OC(O)C(O)C(O)C1O",
    "thalidomide":      "O=C1CCC(=O)N1C1CCC(=O)N(C1=O)c1ccccc1",
    "methotrexate":     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O",
    "tetracycline":     "CN(C)C1C(O)=C(C(N)=O)C(=O)c2c(O)cccc21",
    "ciprofloxacin":    "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "vitamin c":        "OCC1OC(=O)C(O)=C1O",
    "diazepam":         "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "fluoxetine":       "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "sertraline":       "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "atorvastatin":     "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "metoprolol":       "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "tamoxifen":        "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "imatinib":         "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "codeine":          "COc1ccc2CC3N(C)CCC3=Cc2c1O",
    "fentanyl":         "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1",
    "tramadol":         "OC1(CCCCC1)C(CN(C)C)c1ccccc1",
    "acyclovir":        "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
    "fluconazole":      "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
    "testosterone":     "CC12CCC3c4ccc(O)cc4CCC3C1CCC2=O",
    "estradiol":        "OC1=CC2=C(CC[C@@H]3[C@@H]2CC[C@]4(C)[C@@H]3CC[C@@H]4O)C=C1",
    "cortisol":         "OCC(=O)C1(O)CCC2C3CCC4=CC(=O)CCC4(C)C3C(O)CC21C",
    "dexamethasone":    "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)(C(=O)CO)C1O",
    "cholesterol":      "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
    "benzene":          "c1ccccc1",
    "nitrobenzene":     "O=[N+]([O-])c1ccccc1",
    "aniline":          "Nc1ccccc1",
    "capsaicin":        "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "resveratrol":      "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
    "curcumin":         "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
    "penicillin g":     "CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1",
    "viagra":           "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "oseltamivir":      "CCOC(=O)C1=C(OC(CC)CC)CC(NC(C)=O)CC1N",
    "remdesivir":       "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(O)C(C(N2C=CC(=O)NC2=O)O1)O)Oc1ccccc1",
    "doxorubicin":      "COc1cccc2C(=O)c3c(O)c4C[C@@](O)(CC(=O)CO)C[C@H](O[C@H]5C[C@H](N)[C@H](O)[C@H](C)O5)[C@@H]4c(O)c3C(=O)c12",
    "cyclophosphamide": "ClCCN(CCCl)P1(=O)NCCCO1",
    "aflatoxin b1":     "O=c1occc2c1cc1c(=O)oc3ccoc3c1c2",
}

PRESETS = {
    "💊 Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "☕ Caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "🤕 Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "💉 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "💙 Metformin":   "CN(C)C(=N)NC(=N)N",
    "🩸 Warfarin":    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    with open('results/models.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_original_data():
    return pd.read_csv('data/tox21.csv')

@st.cache_resource
def load_train_fingerprints():
    """Load pre-computed training set RDKit fingerprints for AD checks."""
    try:
        with open('results/train_fingerprints.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Training fingerprints not found. Run 02_features.py first.")
        return None


models      = load_models()
df_original = load_original_data()
train_fps   = load_train_fingerprints()

# ============================================
# HELPER FUNCTIONS
# ============================================
def lookup_pubchem(drug_name):
    name = drug_name.strip()

    # Check local database first
    if name.lower() in DRUG_DATABASE:
        return DRUG_DATABASE[name.lower()]

    # API 1: PubChem
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
        r   = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200:
            return r.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass

    # API 2: OPSIN
    try:
        url = f"https://opsin.ch.cam.ac.uk/opsin/{urllib.parse.quote(name)}.json"
        r   = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200 and 'smiles' in r.json():
            return r.json()['smiles']
    except:
        pass

    # API 3: CIR
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(name)}/smiles"
        r   = requests.get(url, timeout=8, verify=False)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except:
        pass

    return None


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


def get_3d_viewer(mol):
    try:
        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if result == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_block = MolToMolBlock(mol_3d)
        html = f"""
        <div style="height:320px;width:100%;position:relative;">
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <div id="viewer3d"
             style="height:320px;width:100%;
                    background:#0d1117;
                    border-radius:12px;
                    border:1px solid #30363d;">
        </div>
        <script>
            let v = $3Dmol.createViewer(
                document.getElementById('viewer3d'),
                {{backgroundColor:'0x0d1117'}}
            );
            v.addModel(`{mol_block}`,'mol');
            v.setStyle({{}},{{
                stick:{{radius:0.15,colorscheme:'rasmol'}},
                sphere:{{scale:0.25,colorscheme:'rasmol'}}
            }});
            v.zoomTo();
            v.spin(true);
            v.render();
        </script>
        </div>
        """
        return html
    except:
        return None


def plot_radar(predictions):
    targets = list(predictions.keys())
    values  = list(predictions.values())
    targets_loop = targets + [targets[0]]
    values_loop  = values  + [values[0]]

    avg = np.mean(values)
    color = (
        'rgba(231,76,60,0.4)'   if avg > 0.5  else
        'rgba(243,156,18,0.4)'  if avg > 0.3  else
        'rgba(46,204,113,0.4)'
    )
    line_color = (
        'rgb(231,76,60)'   if avg > 0.5  else
        'rgb(243,156,18)'  if avg > 0.3  else
        'rgb(46,204,113)'
    )

    fig = go.Figure(data=go.Scatterpolar(
        r=values_loop,
        theta=targets_loop,
        fill='toself',
        fillcolor=color,
        line=dict(color=line_color, width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                color='#8b949e'
            ),
            angularaxis=dict(color='#8b949e')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11),
        showlegend=False,
        height=420,
        margin=dict(t=40, b=40)
    )
    return fig


def find_similar(query_mol, df, top_n=5):
    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        query_mol, radius=2, nBits=1024
    )
    sims = []
    for i, smi in enumerate(df['smiles']):
        try:
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                continue
            fp  = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            sims.append((i, sim, smi))
        except:
            continue
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[1:top_n+1]


def check_applicability_domain(smiles, train_fps, threshold=0.4):
    """
    Check if a molecule falls within the Applicability Domain
    of the training set using Tanimoto nearest-neighbor similarity.

    Returns (in_domain: bool, max_similarity: float or None).
    """
    if train_fps is None:
        return True, None  # Graceful fallback if fingerprints unavailable

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True, None

        query_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=1024
        )

        # BulkTanimotoSimilarity is C++ optimized — handles thousands fast
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
        max_sim = max(similarities) if similarities else 0.0

        return (max_sim >= threshold), max_sim
    except Exception:
        return True, None  # Don't block predictions on AD errors


def run_prediction(smiles_input, models):
    feat_cols = models[list(models.keys())[0]]['feature_cols']
    result    = get_features(smiles_input, feat_cols)

    if result is None:
        st.error("❌ Could not process this molecule. Check the SMILES string.")
        return

    row, mol, basic_props, tox_flags = result

    # ---- Applicability Domain Check ----
    ad_in_domain, ad_similarity = check_applicability_domain(
        smiles_input, train_fps, threshold=0.4
    )

    if ad_similarity is not None:
        if not ad_in_domain:
            st.warning(
                f"⚠️ **Out of Applicability Domain:** This molecule is highly "
                f"dissimilar to our training data (Nearest Neighbor Similarity: "
                f"**{ad_similarity:.1%}** < 40%). The model is extrapolating. "
                f"Predictions may be less reliable."
            )
        st.info(f"🔬 **Nearest Neighbor Similarity:** {ad_similarity:.1%}")

    # ---- Layout ----
    col1, col2 = st.columns([1, 2])

    with col1:
        # 3D viewer
        viewer_html = get_3d_viewer(mol)
        if viewer_html:
            st.markdown("**🔬 3D Structure — drag to rotate**")
            st.components.v1.html(viewer_html, height=340)
        else:
            st.markdown("**Molecule Structure**")
            img = Draw.MolToImage(mol, size=(280, 220))
            st.image(img)

        # Molecular properties
        st.markdown("**Molecular Properties**")
        props_df = pd.DataFrame(
            basic_props.items(),
            columns=['Property', 'Value']
        )
        props_df['Value'] = props_df['Value'].round(3)
        st.dataframe(props_df, hide_index=True, use_container_width=True)

    with col2:
        # Predictions
        predictions = {}
        for target, model_data in models.items():
            prob = model_data['model'].predict_proba(row)[0][1]
            predictions[target] = prob

        avg_risk = np.mean(list(predictions.values()))

        # Risk banner
        if avg_risk > 0.5:
            st.markdown(
                f'<div class="risk-high">⚠️ Overall Risk Score: {avg_risk:.1%} — HIGH RISK</div>',
                unsafe_allow_html=True
            )
        elif avg_risk > 0.3:
            st.markdown(
                f'<div class="risk-moderate">⚡ Overall Risk Score: {avg_risk:.1%} — MODERATE RISK</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-low">✅ Overall Risk Score: {avg_risk:.1%} — LOW RISK</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar chart
        st.plotly_chart(
            plot_radar(predictions),
            use_container_width=True
        )

    # Per target metrics
    st.markdown("---")
    st.markdown("**📊 Per-Assay Breakdown**")
    pred_cols = st.columns(4)
    for i, (target, prob) in enumerate(predictions.items()):
        with pred_cols[i % 4]:
            icon = "🔴" if prob > 0.5 else "🟡" if prob > 0.3 else "🟢"
            st.metric(label=f"{icon} {target}", value=f"{prob:.1%}")

    # Toxicophore alerts
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
        st.success("✅ No known toxicophores detected.")

    # ADMET
    st.markdown("---")
    st.markdown("**💊 ADMET Drug Safety Profile**")
    mw   = basic_props['MolWt']
    logp = basic_props['LogP']
    hbd  = basic_props['NumHDonors']
    hba  = basic_props['NumHAcceptors']
    tpsa = basic_props['TPSA']

    admet = {
        'Oral Bioavailability':
            '✅ Good'    if mw<500 and logp<5 and hbd<=5 and hba<=10 else '⚠️ Poor',
        'GI Absorption':
            '✅ High'    if tpsa<140 else '⚠️ Low',
        'BBB Penetration':
            '✅ Likely'  if logp>0 and tpsa<90 and mw<450 else '⚠️ Unlikely',
        'hERG Cardiac Risk':
            '⚠️ High'   if logp>3.7 and mw>400 else '✅ Low',
        'Hepatotoxicity Risk':
            '⚠️ High'   if logp>4 else '✅ Low',
        'Lipinski Compliant':
            '✅ Yes'     if mw<=500 and logp<=5 and hbd<=5 and hba<=10 else '⚠️ No',
        'Lead-Like':
            '✅ Yes'     if mw<=350 and logp<=3 and hbd<=3 and hba<=7 else '⚠️ No',
    }

    a1, a2 = st.columns(2)
    items  = list(admet.items())
    with a1:
        for k, v in items[:4]:
            st.markdown(f"**{k}:** {v}")
    with a2:
        for k, v in items[4:]:
            st.markdown(f"**{k}:** {v}")

    # Disclaimer
    st.caption(
        "⚠️ These predictions are based on Tox21 cellular toxicity assays only. "
        "They do not cover addiction, neurotoxicity, overdose risk, or chronic toxicity. "
        "Always consult medical literature for complete safety profiles."
    )

    # Similarity search
    st.markdown("---")
    st.markdown("**🔎 Most Similar Known Molecules in Training Data**")
    similar = find_similar(mol, df_original)
    if similar:
        sim_cols = st.columns(5)
        for col, (idx, sim, smi) in zip(sim_cols, similar):
            sim_mol = Chem.MolFromSmiles(str(smi))
            if sim_mol:
                img = Draw.MolToImage(sim_mol, size=(150, 120))
                with col:
                    st.image(img)
                    st.caption(f"Sim: {sim:.1%}")

    # Download report
    st.markdown("---")
    report = pd.DataFrame({
        'Target':               list(predictions.keys()),
        'Toxicity Probability': [f"{v:.1%}" for v in predictions.values()],
        'Risk Level':           ['HIGH' if v>0.5 else 'MODERATE' if v>0.3 else 'LOW'
                                 for v in predictions.values()]
    })
    st.download_button(
        label="📥 Download Full Report as CSV",
        data=report.to_csv(index=False),
        file_name="toxicity_report.csv",
        mime="text/csv"
    )


# ============================================
# MAIN UI
# ============================================
st.title("🧬 ToxPredict")
st.markdown("### AI-Powered Drug Toxicity Predictor")
st.markdown(
    "Predict toxicity of **any molecule** across **12 biological targets** "
    "with 3D visualization, ADMET profiling, and molecular explainability."
)
st.markdown("---")

# Mode selector
mode = st.radio(
    "Select Mode:",
    ["🔬 Single Drug Analysis", "⚖️ Compare Two Drugs", "📋 Batch Screening"],
    horizontal=True
)

st.markdown("---")

# ============================================
# MODE 1: SINGLE DRUG
# ============================================
if mode == "🔬 Single Drug Analysis":

    # Initialize session state
    if 'current_smiles' not in st.session_state:
        st.session_state['current_smiles'] = "CC(=O)Oc1ccccc1C(=O)O"

    # Preset buttons
    st.markdown("**Quick test with real drugs:**")
    btn_cols = st.columns(6)
    for col, (label, smi) in zip(btn_cols, PRESETS.items()):
        if col.button(label):
            st.session_state['current_smiles'] = smi

    st.markdown("---")

    # Input tabs
    tab1, tab2 = st.tabs(["🔤 Search by Drug Name", "🧪 Enter SMILES Manually"])

    with tab1:
        st.markdown("Type any drug name — fetched from PubChem automatically.")
        drug_name_input = st.text_input(
            "Drug name:",
            placeholder="e.g. Thalidomide, Doxorubicin, Estradiol, Fentanyl..."
        )
        if st.button("🔍 Find Drug", key="search_btn"):
            if drug_name_input.strip() == "":
                st.warning("Please enter a drug name.")
            else:
                with st.spinner(f"Looking up '{drug_name_input}'..."):
                    found_smiles = lookup_pubchem(drug_name_input)
                if found_smiles:
                    st.session_state['current_smiles'] = found_smiles
                    st.success(f"✅ Found **{drug_name_input}**")
                    st.code(found_smiles, language="text")
                else:
                    st.error(f"❌ Could not find **'{drug_name_input}'**. Try the SMILES tab.")

    with tab2:
        st.markdown("Paste a SMILES string directly.")
        manual_smiles = st.text_input(
            "SMILES String:",
            value=st.session_state['current_smiles'],
        )
        if st.button("Load SMILES", key="load_btn"):
            st.session_state['current_smiles'] = manual_smiles

    st.markdown("---")
    st.info(f"**Loaded:** `{st.session_state['current_smiles']}`")

    if st.button("🔬 Predict Toxicity", type="primary", key="predict_single"):
        run_prediction(st.session_state['current_smiles'], models)

# ============================================
# MODE 2: COMPARE TWO DRUGS
# ============================================
elif mode == "⚖️ Compare Two Drugs":

    st.markdown("### Compare toxicity profiles of two drugs side by side")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Drug A")
        drug_a = st.text_input("Drug A name:", placeholder="e.g. Aspirin")
        if st.button("Find Drug A"):
            smiles_a = lookup_pubchem(drug_a)
            if smiles_a:
                st.session_state['drug_a_smiles'] = smiles_a
                st.success(f"✅ {drug_a}")
                st.code(smiles_a)
            else:
                st.error("Not found")

    with col_b:
        st.markdown("#### Drug B")
        drug_b = st.text_input("Drug B name:", placeholder="e.g. Ibuprofen")
        if st.button("Find Drug B"):
            smiles_b = lookup_pubchem(drug_b)
            if smiles_b:
                st.session_state['drug_b_smiles'] = smiles_b
                st.success(f"✅ {drug_b}")
                st.code(smiles_b)
            else:
                st.error("Not found")

    if st.button("⚖️ Compare Now", type="primary"):
        if 'drug_a_smiles' not in st.session_state or 'drug_b_smiles' not in st.session_state:
            st.warning("Please find both drugs first.")
        else:
            feat_cols = models[list(models.keys())[0]]['feature_cols']

            result_a = get_features(st.session_state['drug_a_smiles'], feat_cols)
            result_b = get_features(st.session_state['drug_b_smiles'], feat_cols)

            if result_a and result_b:
                row_a, mol_a, _, _ = result_a
                row_b, mol_b, _, _ = result_b

                pred_a = {t: models[t]['model'].predict_proba(row_a)[0][1] for t in models}
                pred_b = {t: models[t]['model'].predict_proba(row_b)[0][1] for t in models}

                # Side by side structures
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.markdown(f"**Drug A: {drug_a}**")
                    st.image(Draw.MolToImage(mol_a, size=(250, 200)))
                    st.metric("Overall Risk", f"{np.mean(list(pred_a.values())):.1%}")
                with img_col2:
                    st.markdown(f"**Drug B: {drug_b}**")
                    st.image(Draw.MolToImage(mol_b, size=(250, 200)))
                    st.metric("Overall Risk", f"{np.mean(list(pred_b.values())):.1%}")

                # Radar comparison
                r1, r2 = st.columns(2)
                with r1:
                    st.plotly_chart(plot_radar(pred_a), use_container_width=True)
                with r2:
                    st.plotly_chart(plot_radar(pred_b), use_container_width=True)

                # Comparison table
                st.markdown("**Detailed Comparison**")
                compare_df = pd.DataFrame({
                    f'{drug_a}':    {t: f"{v:.1%}" for t, v in pred_a.items()},
                    f'{drug_b}':    {t: f"{v:.1%}" for t, v in pred_b.items()},
                    'Higher Risk':  {
                        t: drug_a if pred_a[t] > pred_b[t] else drug_b
                        for t in pred_a
                    }
                })
                st.dataframe(compare_df, use_container_width=True)

                # Download
                st.download_button(
                    "📥 Download Comparison",
                    compare_df.to_csv(),
                    "comparison.csv",
                    "text/csv"
                )

# ============================================
# MODE 3: BATCH SCREENING
# ============================================
elif mode == "📋 Batch Screening":

    st.markdown("### Screen multiple molecules at once")
    st.markdown("Upload a CSV file with a `smiles` column.")

    # Example download
    example = pd.DataFrame({'smiles': [
        'CC(=O)Oc1ccccc1C(=O)O',
        'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    ]})
    st.download_button(
        "📥 Download Example CSV",
        example.to_csv(index=False),
        "example_molecules.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV:", type=['csv'])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)

        if 'smiles' not in batch_df.columns:
            st.error("❌ CSV must have a 'smiles' column.")
        else:
            st.success(f"✅ Loaded {len(batch_df)} molecules")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🔬 Screen All Molecules", type="primary"):
                feat_cols = models[list(models.keys())[0]]['feature_cols']
                results   = []
                progress  = st.progress(0)
                status    = st.empty()

                for i, row in batch_df.iterrows():
                    status.text(f"Processing molecule {i+1}/{len(batch_df)}...")
                    result = get_features(str(row['smiles']), feat_cols)
                    if result:
                        row_feats, _, _, _ = result
                        preds = {
                            t: round(models[t]['model'].predict_proba(row_feats)[0][1], 3)
                            for t in models
                        }
                        preds['smiles']       = row['smiles']
                        preds['overall_risk'] = round(np.mean(list(preds.values())[:-1]), 3)
                        preds['risk_level']   = (
                            'HIGH'     if preds['overall_risk'] > 0.5 else
                            'MODERATE' if preds['overall_risk'] > 0.3 else
                            'LOW'
                        )
                        results.append(preds)
                    progress.progress((i+1) / len(batch_df))

                status.empty()
                results_df = pd.DataFrame(results)

                st.success(f"✅ Screened {len(results_df)} molecules!")

                # Summary stats
                m1, m2, m3 = st.columns(3)
                m1.metric("High Risk",     f"{(results_df['risk_level']=='HIGH').sum()}")
                m2.metric("Moderate Risk", f"{(results_df['risk_level']=='MODERATE').sum()}")
                m3.metric("Low Risk",      f"{(results_df['risk_level']=='LOW').sum()}")

                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    "📥 Download Results",
                    results_df.to_csv(index=False),
                    "batch_results.csv",
                    "text/csv"
                )