# ============================================
# TOXPREDICT — AI Drug Toxicity Predictor
# Complete Final Version with all features
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
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
except:
    requests = None

sys.path.append('..')

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Draw, Descriptors, rdMolDescriptors,
    AllChem, MolFromSmarts, MolToMolBlock, MACCSkeys
)
from rdkit.Chem.Draw import rdMolDraw2D

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
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
    }
    .risk-high {
        background: #3d1515;
        border: 1px solid #e74c3c;
        border-radius: 10px;
        padding: 15px;
        color: #e74c3c;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .risk-moderate {
        background: #3d2e10;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 15px;
        color: #f39c12;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .risk-low {
        background: #0f3d20;
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 15px;
        color: #2ecc71;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .confidence-high   { color: #2ecc71; font-size: 12px; }
    .confidence-medium { color: #f39c12; font-size: 12px; }
    .confidence-low    { color: #e74c3c; font-size: 12px; }
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

TOXICOPHORE_DESCRIPTIONS = {
    'tox_Nitro_group':      'DNA damage & mutagenicity',
    'tox_Aldehyde':         'Protein binding & cell damage',
    'tox_Epoxide':          'DNA alkylation & carcinogenicity',
    'tox_Aromatic_amine':   'Metabolic activation & carcinogenicity',
    'tox_Hydrazine':        'Hepatotoxicity & carcinogenicity',
    'tox_Alkyl_halide':     'Reactive electrophile',
    'tox_Michael_acceptor': 'Covalent protein binding',
    'tox_Quinone':          'Oxidative stress & redox cycling',
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
    "estradiol":        "OC1=CC2=C(CCC3C2CCC4(C)C3CCC4O)C=C1",
    "cortisol":         "OCC(=O)C1(O)CCC2C3CCC4=CC(=O)CCC4(C)C3C(O)CC21C",
    "dexamethasone":    "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)(C(=O)CO)C1O",
    "cholesterol":      "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
    "benzene":          "c1ccccc1",
    "nitrobenzene":     "O=[N+]([O-])c1ccccc1",
    "aniline":          "Nc1ccccc1",
    "capsaicin":        "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "resveratrol":      "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
    "curcumin":         "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
    "viagra":           "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "oseltamivir":      "CCOC(=O)C1=C(OC(CC)CC)CC(NC(C)=O)CC1N",
    "remdesivir":       "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(O)C(C(N2C=CC(=O)NC2=O)O1)O)Oc1ccccc1",
    "doxorubicin":      "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)CC4c(O)c3C(=O)c12",
    "cyclophosphamide": "ClCCN(CCCl)P1(=O)NCCCO1",
    "aflatoxin b1":     "O=c1occc2c1cc1c(=O)oc3ccoc3c1c2",
    "penicillin g":     "CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1",
    "methotrexate":     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O",
}

PRESETS = {
    "💊 Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "☕ Caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "🤕 Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "💉 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "💙 Metformin":   "CN(C)C(=N)NC(=N)N",
    "🩸 Warfarin":    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}

DRUG_DISASTERS = {
    "⚠️ Thalidomide (1957)": {
        "event": "Caused 10,000+ birth defects worldwide",
        "lesson": "Teratogenicity not detected by available tests",
        "note": "Low Tox21 score — same gap exists today"
    },
    "⚠️ Vioxx (2004)": {
        "event": "Withdrawn — linked to 60,000+ cardiac deaths",
        "lesson": "Cardiovascular risk missed in clinical trials",
        "note": "hERG cardiac risk flagged in our ADMET profile"
    },
    "⚠️ Fen-Phen (1997)": {
        "event": "Withdrawn — caused heart valve damage",
        "lesson": "Long-term organ toxicity undetected short-term",
        "note": "SR-MMP mitochondrial flag as warning signal"
    },
    "⚠️ Chloroform (1900s)": {
        "event": "Early anesthetic — caused liver failure & deaths",
        "lesson": "Hepatotoxicity underestimated at therapeutic doses",
        "note": "High LogP hepatotoxicity flag in ADMET"
    },
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

@st.cache_data
def load_metrics():
    return pd.read_csv('results/metrics.csv', index_col=0)

models      = load_models()
df_original = load_original_data()

# ============================================
# HELPER FUNCTIONS
# ============================================

def lookup_drug(drug_name):
    """Local DB first, then 3 API fallbacks"""
    key = drug_name.strip().lower()
    if key in DRUG_DATABASE:
        return DRUG_DATABASE[key]

    encoded = urllib.parse.quote(drug_name.strip())

    # API 1: PubChem
    try:
        if requests:
            r = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON",
                timeout=8, verify=False
            )
            if r.status_code == 200:
                return r.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
        else:
            req  = urllib.request.urlopen(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON",
                timeout=8
            )
            data = json.loads(req.read().decode())
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass

    # API 2: OPSIN
    try:
        if requests:
            r = requests.get(f"https://opsin.ch.cam.ac.uk/opsin/{encoded}.json", timeout=8, verify=False)
            if r.status_code == 200 and 'smiles' in r.json():
                return r.json()['smiles']
    except:
        pass

    # API 3: CIR
    try:
        if requests:
            r = requests.get(
                f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles",
                timeout=8, verify=False
            )
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
    except:
        pass

    return None


def get_features(smiles, feat_cols):
    """Extract all features from SMILES"""
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

        mfp_size   = len([c for c in feat_cols if c.startswith('MFP_')])
        maccs_size = len([c for c in feat_cols if c.startswith('MACCS_')])

        fp      = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=max(mfp_size, 1024))
        fp_dict = {f'MFP_{i}': int(fp[i]) for i in range(mfp_size)}

        maccs_fp   = MACCSkeys.GenMACCSKeys(mol)
        maccs_dict = {f'MACCS_{i}': int(maccs_fp[i]) for i in range(min(maccs_size, 167))}

        all_feats = {**basic, **fp_dict, **maccs_dict, **tox_flags}
        row       = pd.DataFrame([all_feats]).reindex(columns=feat_cols, fill_value=0)

        return row, mol, basic, tox_flags
    except:
        return None


def get_confidence(prob):
    dist = abs(prob - 0.5) * 2
    if dist > 0.7:
        return "High Confidence", "🟢"
    elif dist > 0.4:
        return "Moderate Confidence", "🟡"
    else:
        return "Low Confidence", "🔴"


def draw_highlighted_mol(mol, active_smarts):
    """Draw molecule with toxic substructures highlighted in red"""
    try:
        highlight_atoms = set()
        highlight_bonds = set()

        for smarts in active_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                highlight_atoms.update(match)
                for bond in mol.GetBonds():
                    if (bond.GetBeginAtomIdx() in match and
                            bond.GetEndAtomIdx() in match):
                        highlight_bonds.add(bond.GetIdx())

        drawer = rdMolDraw2D.MolDraw2DSVG(320, 260)
        if highlight_atoms:
            atom_colors = {i: (0.9, 0.2, 0.2) for i in highlight_atoms}
            bond_colors = {i: (0.9, 0.2, 0.2) for i in highlight_bonds}
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_atoms),
                highlightAtomColors=atom_colors,
                highlightBonds=list(highlight_bonds),
                highlightBondColors=bond_colors
            )
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except:
        return None


def get_3d_viewer(mol):
    """Generate 3D rotating molecule viewer HTML"""
    try:
        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if result == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_block = MolToMolBlock(mol_3d)
        html = f"""
        <div style="height:300px;width:100%;position:relative;">
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <div id="viewer3d"
             style="height:300px;width:100%;
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


def plot_radar(predictions, title=""):
    """Radar chart for toxicity profile"""
    targets      = list(predictions.keys())
    values       = list(predictions.values())
    targets_loop = targets + [targets[0]]
    values_loop  = values  + [values[0]]
    avg          = np.mean(values)

    color      = ('rgba(231,76,60,0.4)'  if avg > 0.5 else
                  'rgba(243,156,18,0.4)' if avg > 0.3 else
                  'rgba(46,204,113,0.4)')
    line_color = ('rgb(231,76,60)'  if avg > 0.5 else
                  'rgb(243,156,18)' if avg > 0.3 else
                  'rgb(46,204,113)')

    fig = go.Figure(data=go.Scatterpolar(
        r=values_loop,
        theta=targets_loop,
        fill='toself',
        fillcolor=color,
        line=dict(color=line_color, width=2),
        name='Risk'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickformat='.0%', color='#8b949e',
                gridcolor='#30363d'
            ),
            angularaxis=dict(color='#8b949e', gridcolor='#30363d')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11),
        showlegend=False,
        height=400,
        margin=dict(t=50, b=40),
        title=dict(text=title, font=dict(size=13))
    )
    return fig


def find_similar(query_mol, df, top_n=5):
    """Tanimoto similarity search"""
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


def suggest_safer_alternatives(mol, df, tox_flags):
    """Find similar molecules without the detected toxicophores"""
    active_tox = [
        k for k, v in tox_flags.items()
        if v == 1 and k != 'total_toxicophores'
    ]
    if not active_tox:
        return []

    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=1024
    )
    safer = []
    for smi in df['smiles']:
        try:
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                continue
            fp  = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            if sim < 0.3 or sim >= 1.0:
                continue
            has_tox = any(
                MolFromSmarts(TOXICOPHORES[name]) and
                m.HasSubstructMatch(MolFromSmarts(TOXICOPHORES[name]))
                for name in active_tox
                if name in TOXICOPHORES
            )
            if not has_tox:
                safer.append((sim, smi, m))
        except:
            continue
    safer.sort(key=lambda x: x[0], reverse=True)
    return safer[:3]


def run_prediction(smiles_input, models, show_alternatives=True):
    """Full prediction pipeline"""
    feat_cols = models[list(models.keys())[0]]['feature_cols']
    result    = get_features(smiles_input, feat_cols)

    if result is None:
        st.error("❌ Could not process this molecule. Check the SMILES string.")
        return

    row, mol, basic_props, tox_flags = result

    # Predictions
    predictions = {}
    for target, model_data in models.items():
        prob = model_data['model'].predict_proba(row)[0][1]
        predictions[target] = prob

    avg_risk = np.mean(list(predictions.values()))

    # ---- Top layout ----
    col1, col2 = st.columns([1, 2])

    with col1:
        # Active toxicophore SMARTS for highlighting
        active_smarts = [
            smarts for name, smarts in TOXICOPHORES.items()
            if tox_flags.get(name, 0) == 1
        ]

        # Try SVG highlighted drawing first
        svg = draw_highlighted_mol(mol, active_smarts)
        if svg and active_smarts:
            st.markdown("**🔬 Structure (toxic atoms in red)**")
            st.components.v1.html(
                f"""
                <div style="background:#1a1a2e; border-radius:10px; 
                            padding:10px; border:1px solid #30363d;">
                    {svg}
                </div>
                """,
                height=290
            )
        else:
            # Try 3D viewer
            viewer_html = get_3d_viewer(mol)
            if viewer_html:
                st.markdown("**🔬 3D Structure — drag to rotate**")
                st.components.v1.html(viewer_html, height=320)
            else:
                st.markdown("**🔬 Molecule Structure**")
                img = Draw.MolToImage(mol, size=(280, 220))
                st.image(img)

        # Molecular properties table
        st.markdown("**📐 Molecular Properties**")
        props_df = pd.DataFrame(
            basic_props.items(),
            columns=['Property', 'Value']
        )
        props_df['Value'] = props_df['Value'].round(3)
        st.dataframe(props_df, hide_index=True, use_container_width=True)

    with col2:
        # Risk banner
        if avg_risk > 0.5:
            st.markdown(
                f'<div class="risk-high">⚠️ Overall Risk: {avg_risk:.1%} — HIGH RISK</div>',
                unsafe_allow_html=True
            )
        elif avg_risk > 0.3:
            st.markdown(
                f'<div class="risk-moderate">⚡ Overall Risk: {avg_risk:.1%} — MODERATE RISK</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-low">✅ Overall Risk: {avg_risk:.1%} — LOW RISK</div>',
                unsafe_allow_html=True
            )

        # Radar chart
        st.plotly_chart(
            plot_radar(predictions, "Toxicity Profile — All 12 Assays"),
            use_container_width=True
        )

    # Per assay breakdown
    st.markdown("---")
    st.markdown("**📊 Per-Assay Breakdown with Confidence**")
    pred_cols = st.columns(4)
    for i, (target, prob) in enumerate(predictions.items()):
        conf_label, conf_icon = get_confidence(prob)
        with pred_cols[i % 4]:
            risk_icon = "🔴" if prob > 0.5 else "🟡" if prob > 0.3 else "🟢"
            st.metric(
                label=f"{risk_icon} {target}",
                value=f"{prob:.1%}",
                delta=f"{conf_icon} {conf_label}"
            )

    # Toxicophore alerts
    st.markdown("---")
    st.markdown("**🔍 Toxicophore Analysis**")
    found = {
        k: v for k, v in tox_flags.items()
        if v == 1 and k != 'total_toxicophores'
    }
    if found:
        st.warning(f"⚠️ Found **{len(found)} toxicophore(s)** in this molecule:")
        for name in found:
            clean_name = name.replace('tox_', '').replace('_', ' ')
            desc       = TOXICOPHORE_DESCRIPTIONS.get(name, '')
            st.markdown(f"  - 🔴 **{clean_name}** — {desc}")
    else:
        st.success("✅ No known toxicophores detected in this molecule.")

    # ADMET Profile
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
        "⚠️ Predictions based on Tox21 cellular assays only. "
        "Does not cover addiction, neurotoxicity, overdose risk, or chronic toxicity. "
        "Consult medical literature for complete safety profiles."
    )

    # Safer alternatives
    if show_alternatives and found:
        st.markdown("---")
        st.markdown("**💡 Structurally Similar Molecules Without Detected Toxicophores**")
        alternatives = suggest_safer_alternatives(mol, df_original, tox_flags)
        if alternatives:
            alt_cols = st.columns(3)
            for col, (sim, smi, alt_mol) in zip(alt_cols, alternatives):
                with col:
                    alt_img = Draw.MolToImage(alt_mol, size=(180, 140))
                    st.image(alt_img)
                    st.caption(f"Similarity: {sim:.1%} | No toxicophores detected")
        else:
            st.info("No safer alternatives found in training dataset.")

    # Similarity search
    st.markdown("---")
    st.markdown("**🔎 Most Similar Molecules in Training Data**")
    similar = find_similar(mol, df_original)
    if similar:
        sim_cols = st.columns(5)
        for col, (idx, sim, smi) in zip(sim_cols, similar):
            sim_mol = Chem.MolFromSmiles(str(smi))
            if sim_mol:
                with col:
                    st.image(Draw.MolToImage(sim_mol, size=(140, 110)))
                    st.caption(f"Sim: {sim:.1%}")

    # Download
    st.markdown("---")
    report = pd.DataFrame({
        'Target':               list(predictions.keys()),
        'Toxicity Probability': [f"{v:.1%}" for v in predictions.values()],
        'Risk Level':           ['HIGH' if v>0.5 else 'MODERATE' if v>0.3 else 'LOW'
                                 for v in predictions.values()],
        'Confidence':           [get_confidence(v)[0] for v in predictions.values()]
    })
    st.download_button(
        label="📥 Download Full Toxicity Report (CSV)",
        data=report.to_csv(index=False),
        file_name="toxicity_report.csv",
        mime="text/csv"
    )


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🧬 ToxPredict")
    st.markdown("*AI Drug Toxicity Predictor*")
    st.markdown("---")

    st.markdown("### 📊 Model Stats")
    try:
        metrics_df = load_metrics()
        st.metric("Best AUC",      f"{metrics_df['ROC-AUC'].max():.4f}")
        st.metric("Mean AUC",      f"{metrics_df['ROC-AUC'].mean():.4f}")
        st.metric("Targets > 0.75",f"{(metrics_df['ROC-AUC']>0.75).sum()}/12")
        st.metric("Features Used", "2200+")
    except:
        st.info("Run training first.")

    st.markdown("---")
    st.markdown("### 📚 Drug Disaster Timeline")
    st.caption("Why toxicity screening matters:")
    for drug, info in DRUG_DISASTERS.items():
        with st.expander(drug):
            st.markdown(f"**What happened:** {info['event']}")
            st.markdown(f"**Lesson:** {info['lesson']}")
            st.markdown(f"**Our model:** {info['note']}")

    st.markdown("---")
    st.markdown("### 🔬 About")
    st.caption(
        "Built for CodeCure Biohackathon | IIT BHU | Track A\n\n"
        "Ensemble: XGBoost + Random Forest + Logistic Regression\n\n"
        "Dataset: Tox21 (7,831 compounds, 12 assays)"
    )

# ============================================
# MAIN UI
# ============================================
st.title("🧬 ToxPredict")
st.markdown("### AI-Powered Drug Toxicity Predictor")
st.markdown(
    "Predict toxicity of **any molecule** across **12 biological targets** — "
    "with 3D visualization, toxicophore highlighting, ADMET profiling, and molecular explainability."
)
st.markdown("---")

# Mode selector
mode = st.radio(
    "Select Mode:",
    [
        "🔬 Single Drug Analysis",
        "⚖️ Compare Two Drugs",
        "📋 Batch Screening",
        "📊 Model Performance"
    ],
    horizontal=True
)

st.markdown("---")

# ============================================
# MODE 1: SINGLE DRUG
# ============================================
if mode == "🔬 Single Drug Analysis":

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
        st.markdown("Type any drug name — fetched automatically from PubChem.")
        drug_name_input = st.text_input(
            "Drug name:",
            placeholder="e.g. Thalidomide, Doxorubicin, Estradiol, Fentanyl..."
        )
        if st.button("🔍 Find Drug", key="search_btn"):
            if not drug_name_input.strip():
                st.warning("Please enter a drug name.")
            else:
                with st.spinner(f"Looking up '{drug_name_input}'..."):
                    found = lookup_drug(drug_name_input)
                if found:
                    st.session_state['current_smiles'] = found
                    st.success(f"✅ Found **{drug_name_input}**")
                    st.code(found)
                else:
                    st.error(f"❌ Could not find **'{drug_name_input}'**.")
                    st.markdown("Try the SMILES tab or check the spelling.")

    with tab2:
        st.markdown("Paste a SMILES string directly.")
        manual = st.text_input("SMILES String:", value=st.session_state['current_smiles'])
        if st.button("Load SMILES", key="load_btn"):
            st.session_state['current_smiles'] = manual

    st.markdown("---")
    st.info(f"**Loaded:** `{st.session_state['current_smiles']}`")

    if st.button("🔬 Predict Toxicity", type="primary", key="predict_single"):
        run_prediction(st.session_state['current_smiles'], models)

# ============================================
# MODE 2: COMPARE TWO DRUGS
# ============================================
elif mode == "⚖️ Compare Two Drugs":

    st.markdown("### ⚖️ Side-by-Side Toxicity Comparison")
    st.markdown("Compare toxicity radar profiles of two molecules.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Drug A")
        drug_a = st.text_input("Drug A:", placeholder="e.g. Estradiol")
        if st.button("Find A"):
            s = lookup_drug(drug_a)
            if s:
                st.session_state['drug_a'] = s
                st.success(f"✅ {drug_a}")
                st.code(s)
            else:
                st.error("Not found")

    with col_b:
        st.markdown("#### Drug B")
        drug_b = st.text_input("Drug B:", placeholder="e.g. Testosterone")
        if st.button("Find B"):
            s = lookup_drug(drug_b)
            if s:
                st.session_state['drug_b'] = s
                st.success(f"✅ {drug_b}")
                st.code(s)
            else:
                st.error("Not found")

    if st.button("⚖️ Compare Now", type="primary"):
        if 'drug_a' not in st.session_state or 'drug_b' not in st.session_state:
            st.warning("Please find both drugs first.")
        else:
            feat_cols = models[list(models.keys())[0]]['feature_cols']
            res_a     = get_features(st.session_state['drug_a'], feat_cols)
            res_b     = get_features(st.session_state['drug_b'], feat_cols)

            if res_a and res_b:
                row_a, mol_a, _, _ = res_a
                row_b, mol_b, _, _ = res_b

                pred_a = {t: models[t]['model'].predict_proba(row_a)[0][1] for t in models}
                pred_b = {t: models[t]['model'].predict_proba(row_b)[0][1] for t in models}

                # Structures
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown(f"**{drug_a}**")
                    st.image(Draw.MolToImage(mol_a, size=(250, 200)))
                    avg_a = np.mean(list(pred_a.values()))
                    color_a = "🔴" if avg_a>0.5 else "🟡" if avg_a>0.3 else "🟢"
                    st.metric("Overall Risk", f"{color_a} {avg_a:.1%}")
                with ic2:
                    st.markdown(f"**{drug_b}**")
                    st.image(Draw.MolToImage(mol_b, size=(250, 200)))
                    avg_b = np.mean(list(pred_b.values()))
                    color_b = "🔴" if avg_b>0.5 else "🟡" if avg_b>0.3 else "🟢"
                    st.metric("Overall Risk", f"{color_b} {avg_b:.1%}")

                # Radar charts
                r1, r2 = st.columns(2)
                with r1:
                    st.plotly_chart(plot_radar(pred_a, drug_a), use_container_width=True)
                with r2:
                    st.plotly_chart(plot_radar(pred_b, drug_b), use_container_width=True)

                # Comparison table
                st.markdown("**Detailed Assay Comparison**")
                compare_df = pd.DataFrame({
                    drug_a:       {t: f"{v:.1%}" for t, v in pred_a.items()},
                    drug_b:       {t: f"{v:.1%}" for t, v in pred_b.items()},
                    'Higher Risk': {
                        t: f"{'🔴 '+drug_a if pred_a[t]>pred_b[t] else '🔴 '+drug_b}"
                        for t in pred_a
                    }
                })
                st.dataframe(compare_df, use_container_width=True)

                st.download_button(
                    "📥 Download Comparison CSV",
                    compare_df.to_csv(),
                    "comparison.csv",
                    "text/csv"
                )

# ============================================
# MODE 3: BATCH SCREENING
# ============================================
elif mode == "📋 Batch Screening":

    st.markdown("### 📋 Batch Molecule Screening")
    st.markdown("Upload a CSV with a `smiles` column to screen multiple molecules at once.")

    example = pd.DataFrame({'smiles': [
        'CC(=O)Oc1ccccc1C(=O)O',
        'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'CC(=O)Nc1ccc(O)cc1',
    ], 'name': ['Aspirin', 'Caffeine', 'Ibuprofen', 'Paracetamol']})

    st.download_button(
        "📥 Download Example CSV Template",
        example.to_csv(index=False),
        "example_molecules.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Upload CSV:", type=['csv'])

    if uploaded:
        batch_df = pd.read_csv(uploaded)

        if 'smiles' not in batch_df.columns:
            st.error("❌ CSV must have a 'smiles' column.")
        else:
            st.success(f"✅ Loaded {len(batch_df)} molecules")
            st.dataframe(batch_df.head(3), use_container_width=True)

            if st.button("🔬 Screen All Molecules", type="primary"):
                feat_cols = models[list(models.keys())[0]]['feature_cols']
                results   = []
                progress  = st.progress(0)
                status    = st.empty()

                for i, row in batch_df.iterrows():
                    status.text(f"Processing {i+1}/{len(batch_df)}...")
                    result = get_features(str(row['smiles']), feat_cols)
                    if result:
                        row_feats, _, _, tox_f = result
                        preds = {
                            t: round(models[t]['model'].predict_proba(row_feats)[0][1], 3)
                            for t in models
                        }
                        preds['smiles']           = row['smiles']
                        preds['name']             = row.get('name', f'Molecule_{i+1}')
                        preds['overall_risk']     = round(np.mean(list(preds.values())[:-2]), 3)
                        preds['risk_level']       = (
                            'HIGH'     if preds['overall_risk'] > 0.5 else
                            'MODERATE' if preds['overall_risk'] > 0.3 else
                            'LOW'
                        )
                        preds['toxicophores_found'] = int(tox_f.get('total_toxicophores', 0))
                        results.append(preds)
                    progress.progress((i+1)/len(batch_df))

                status.empty()
                results_df = pd.DataFrame(results)

                st.success(f"✅ Screened {len(results_df)} molecules!")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔴 High Risk",     f"{(results_df['risk_level']=='HIGH').sum()}")
                m2.metric("🟡 Moderate Risk", f"{(results_df['risk_level']=='MODERATE').sum()}")
                m3.metric("🟢 Low Risk",      f"{(results_df['risk_level']=='LOW').sum()}")
                m4.metric("⚠️ With Toxicophores", f"{(results_df['toxicophores_found']>0).sum()}")

                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    "📥 Download Batch Results CSV",
                    results_df.to_csv(index=False),
                    "batch_results.csv",
                    "text/csv"
                )

# ============================================
# MODE 4: MODEL PERFORMANCE
# ============================================
elif mode == "📊 Model Performance":

    st.markdown("### 📊 Model Performance Dashboard")

    try:
        metrics_df = load_metrics()

        # Summary cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Best AUC",      f"{metrics_df['ROC-AUC'].max():.4f}",
                  delta=metrics_df['ROC-AUC'].idxmax())
        m2.metric("📈 Mean AUC",      f"{metrics_df['ROC-AUC'].mean():.4f}")
        m3.metric("✅ Above 0.75",    f"{(metrics_df['ROC-AUC']>0.75).sum()}/12 targets")
        m4.metric("🔬 Total Features","2200+")

        st.markdown("---")

        # AUC bar chart
        st.markdown("**ROC-AUC Per Assay**")
        try:
            st.image('results/01_per_assay_auc.png', use_column_width=True)
        except:
            st.info("Run 04_visualize.py to generate charts.")

        # Metrics table
        st.markdown("**Detailed Results**")
        display_cols = [c for c in ['ROC-AUC', 'F1', 'PR-AUC', 'CV-AUC', 'CV-Std'] if c in metrics_df.columns]
        st.dataframe(
            metrics_df[display_cols].style.format("{:.4f}").background_gradient(
                subset=['ROC-AUC'], cmap='RdYlGn', vmin=0.5, vmax=1.0
            ),
            use_container_width=True
        )

        st.markdown("---")

        # ROC curves
        st.markdown("**ROC Curves — All 12 Assays**")
        try:
            st.image('results/06_roc_curves.png', use_column_width=True)
        except:
            st.info("Run 04_visualize.py to generate ROC curves.")

        st.markdown("---")

        # SHAP
        st.markdown("**Feature Importance (SHAP)**")
        try:
            st.image('results/04_shap_summary.png', use_column_width=True)
        except:
            st.info("Run 04_visualize.py to generate SHAP plot.")

        st.markdown("---")

        # Heatmap
        st.markdown("**Molecular Properties vs Toxicity Targets**")
        try:
            st.image('results/02_correlation_heatmap.png', use_column_width=True)
        except:
            st.info("Run 04_visualize.py to generate heatmap.")

        st.markdown("---")

        # Real drugs
        st.markdown("**Real FDA Drug Validation**")
        try:
            st.image('results/05_real_drug_predictions.png', use_column_width=True)
        except:
            st.info("Run 04_visualize.py to generate drug predictions.")

    except Exception as e:
        st.error(f"Could not load metrics: {e}")
        st.info("Run the training pipeline first: python src/03_train.py")