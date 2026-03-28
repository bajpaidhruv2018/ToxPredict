# 🧬 ToxPredict — AI-Powered Drug Toxicity Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**CodeCure Biohackathon | Track A — Drug Toxicity Prediction**
**Organized by IIT BHU**

*Predict drug toxicity across 12 biological targets in milliseconds using AI and molecular cheminformatics.*

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [How It Works](#-how-it-works)
- [Results & Visualizations](#-results--visualizations)
- [Limitations & Future Work](#-limitations--future-work)
- [Team](#-team)

---

## 🎯 Problem Statement

Drug development frequently fails due to **unexpected toxicity** — costing billions of dollars and years of research. Early prediction of toxic compounds can:

- Reduce drug development costs significantly
- Improve patient safety
- Accelerate the drug discovery pipeline

Traditional toxicity testing requires expensive lab work, animal testing, and months of analysis. We built an AI system that **predicts toxicity in milliseconds** using only the chemical structure of a molecule.

---

## 💡 Our Solution

**ToxPredict** is an end-to-end AI pipeline that:

1. Takes any drug name or SMILES string as input
2. Extracts 1033+ molecular features using cheminformatics
3. Predicts toxicity across **12 biological assay targets** simultaneously
4. Explains *why* a molecule is toxic using SHAP and toxicophore detection
5. Provides a full **ADMET drug safety profile**
6. Visualizes the molecule in **interactive 3D**

> *"Give us a molecule — we'll tell you if it's toxic and why."*

---

## ⭐ Key Features

### 🧬 Scaffold Splitting (Data Leakage Prevention)
Unlike naive random splits, we use **Murcko scaffold-based splitting** (`GroupShuffleSplit`) to ensure structurally similar molecules never appear in both train and test sets. This prevents artificially inflated metrics and gives scientifically rigorous evaluation.

```
  SMILES → RDKit MurckoScaffoldSmiles → Scaffold Group
      ↓
  GroupShuffleSplit(groups=scaffold)
      ↓
  Train: 6,942 molecules | Test: 881 molecules
  Scaffold Overlap: 0 ✅ (zero leakage)
```

### 🔬 Advanced Feature Engineering
| Feature Type | Count | Description |
|---|---|---|
| Basic Molecular Descriptors | 9 | MolWt, LogP, TPSA, H-donors/acceptors, etc. |
| Morgan Fingerprints | 1024 | Circular molecular fingerprints (radius=2) |
| Toxicophore Flags | 10 | Known toxic substructure detection |
| **Total Features** | **1033+** | Combined feature vector per molecule |

### 🤖 Ensemble Model Architecture
```
Input Molecule
      │
      ▼
┌─────────────────────────────────────┐
│         Feature Extraction          │
│  Basic Descriptors + Morgan FP      │
│  + Toxicophore Detection            │
└─────────────────┬───────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
  XGBoost    Random Forest   Logistic
  (n=200)     (n=200)     Regression
      │           │           │
      └───────────┼───────────┘
                  │
            Soft Voting
                  │
                  ▼
         Toxicity Probability
         (per assay target)
```

### 🧪 Toxicophore Detection
Detects 10 known toxic substructures in real time:
- Nitro groups → DNA damage
- Aldehydes → Protein binding
- Epoxides → DNA alkylation
- Aromatic amines → Carcinogenicity
- Hydrazines → Hepatotoxicity
- Alkyl halides → Reactive intermediates
- Michael acceptors → Electrophilic reactivity
- Quinones → Oxidative stress
- Azo compounds → Metabolic activation
- Peroxides → Oxidative damage

### 🌐 Multi-Source Drug Lookup
Any drug name → SMILES via 3-API fallback chain:
```
Local Database (50+ drugs)
        ↓ if not found
   PubChem API
        ↓ if fails
    OPSIN API
        ↓ if fails
  CIR NCI API
```

### 🖥️ App Modes
- **Single Drug Analysis** — Full toxicity report for any molecule
- **Compare Two Drugs** — Side-by-side radar chart comparison
- **Batch Screening** — Upload CSV, screen hundreds of molecules at once

---

## 📊 Model Performance

### ROC-AUC Scores Across All 12 Tox21 Assays

> **Evaluated using Scaffold Splitting** — all molecules sharing the same Murcko scaffold are strictly in either the train or test set, never both. This eliminates data leakage from structurally similar molecules and provides scientifically honest metrics.

| Assay | Biological Target | ROC-AUC | F1 Score |
|---|---|---|---|
| **NR-AR-LBD** | Androgen receptor (binding) | **0.9480** | 0.8333 |
| **SR-MMP** | Mitochondrial membrane potential | **0.8802** | 0.6127 |
| **NR-AhR** | Aryl hydrocarbon receptor | **0.8587** | 0.4790 |
| **NR-ER-LBD** | Estrogen receptor (binding) | **0.8536** | 0.5645 |
| **NR-AR** | Androgen receptor (full) | **0.8512** | 0.6358 |
| **SR-ATAD5** | DNA damage/replication stress | **0.8142** | 0.2927 |
| **NR-PPAR-gamma** | Metabolic disruption | **0.7883** | 0.2051 |
| **NR-Aromatase** | Estrogen synthesis enzyme | **0.7872** | 0.3404 |
| **NR-ER** | Estrogen receptor (full) | **0.7731** | 0.4916 |
| **SR-ARE** | Oxidative stress response | **0.7384** | 0.4451 |
| **SR-HSE** | Heat shock / stress response | **0.7205** | 0.2453 |
| **SR-p53** | DNA damage response | **0.7183** | 0.2123 |

**✅ 8 out of 12 targets exceed the 0.75 industry benchmark threshold (with scaffold split)**

**📈 Best AUC: 0.9480 (NR-AR-LBD) — achieved with rigorous scaffold-based evaluation**

### Class Imbalance Handling
The Tox21 dataset is heavily imbalanced (far more non-toxic than toxic compounds). We address this via:
- `scale_pos_weight` in XGBoost
- `class_weight='balanced'` in Random Forest and Logistic Regression
- Reporting PR-AUC and F1 alongside ROC-AUC

---

## 🛠️ Tech Stack

### Machine Learning
- `XGBoost` — Gradient boosted trees
- `scikit-learn` — Random Forest, Logistic Regression, VotingClassifier
- `SHAP` — Model explainability

### Cheminformatics
- `RDKit` — Molecular processing, descriptor calculation, Morgan fingerprints, 3D coordinate generation

### Visualization
- `Streamlit` — Web application framework
- `Plotly` — Interactive radar charts
- `3Dmol.js` — 3D molecular viewer (WebGL)
- `Matplotlib` / `Seaborn` — Static plots

### Data
- `Pandas` / `NumPy` — Data processing
- `Tox21 Dataset` — 7,831 compounds, 12 toxicity assay labels

---

## 📁 Project Structure

```
CodeCure/
│
├── data/
│   ├── tox21.csv                    # Raw dataset (7,831 compounds)
│   ├── tox21_train_scaffold.csv     # Scaffold-split train set (6,942)
│   ├── tox21_test_scaffold.csv      # Scaffold-split test set (881)
│   ├── tox21_train_processed.csv    # Train set with 1033+ features
│   └── tox21_test_processed.csv     # Test set with 1033+ features
│
├── src/
│   ├── 01_eda.py                    # Exploratory Data Analysis
│   ├── 01b_scaffold_split.py        # Murcko scaffold splitting (NEW)
│   ├── 02_features.py               # Feature extraction (train/test)
│   ├── 03_train.py                  # Model training (scaffold-split)
│   └── 04_visualize.py              # Generate all result visualizations
│
├── results/
│   ├── models.pkl                   # Saved ensemble models (all 12)
│   ├── metrics.csv                  # AUC + F1 per assay
│   ├── 01_per_assay_auc.png         # AUC bar chart
│   ├── 02_correlation_heatmap.png   # Molecular properties vs toxicity
│   ├── 03_toxicophore_frequency.png # Toxicophore distribution
│   ├── 04_shap_summary.png         # SHAP feature importance
│   └── 05_real_drug_predictions.png # Validation on FDA drugs
│
├── app/
│   └── app.py                       # Streamlit web application
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Windows 10/11 with [Anaconda](https://www.anaconda.com/download) or Miniconda
- GPU recommended (RTX 4050 or similar) — but CPU works fine

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeCure-ToxPredict.git
cd CodeCure-ToxPredict
```

### Step 2 — Create Environment
```bash
conda create -n codecure python=3.10
conda activate codecure
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
Download the Tox21 dataset from [Kaggle](https://www.kaggle.com/datasets/epicskills/tox21-dataset) and place `tox21.csv` in the `data/` folder.

---

## ▶️ How to Run

### Run Full Pipeline (first time)
```bash
conda activate codecure

# Step 1: Explore data
python src/01_eda.py

# Step 2: Scaffold split (prevents data leakage)
python src/01b_scaffold_split.py

# Step 3: Extract features (train & test independently)
python src/02_features.py

# Step 4: Train models (~15 mins)
python src/03_train.py

# Step 5: Generate visualizations
python src/04_visualize.py

# Step 6: Launch app
streamlit run app/app.py
```

### Launch App Only (models already trained)
```bash
conda activate codecure
streamlit run app/app.py
```

App opens at `http://localhost:8501`

---

## 🔬 How It Works

### 1. Molecular Feature Extraction
```
SMILES String → RDKit → 3 Feature Types:

  Type 1: Basic Descriptors (9 features)
    MolWt, LogP, TPSA, H-donors,
    H-acceptors, Rotatable bonds,
    Aromatic rings, Heavy atoms, FractionCSP3

  Type 2: Morgan Fingerprints (1024 features)
    Circular fingerprint encoding
    atomic neighborhoods up to radius=2
    Captures structural patterns at atomic level

  Type 3: Toxicophore Flags (10 features)
    Binary detection of 10 known
    toxic chemical substructures
```

### 2. Ensemble Prediction
```
For each of 12 toxicity targets:
  → Train XGBoost + Random Forest + Logistic Regression
  → Combine via Soft Voting (average probabilities)
  → Handle class imbalance with scale_pos_weight
  → Output: toxicity probability [0.0 - 1.0]
```

### 3. Explainability
```
SHAP TreeExplainer
  → Shows which molecular features
    drove each prediction
  → Beeswarm plot across all samples
  → Force plot for individual molecules
```

### 4. ADMET Profiling
```
From molecular properties → estimate:
  Absorption  → Oral bioavailability, GI absorption
  Distribution → BBB penetration, plasma protein binding
  Metabolism  → CYP interaction risk
  Excretion   → (structural indicators)
  Toxicity    → hERG cardiac risk, hepatotoxicity
```

---

## 📈 Results & Visualizations

### Validation on Real FDA-Approved Drugs

| Drug | Key Finding |
|---|---|
| **Aspirin** | Low risk across all assays ✅ |
| **Estradiol** | 95.9% NR-ER risk — correctly flags estrogen receptor disruption 🔴 |
| **Testosterone** | High NR-AR risk — correctly flags androgen receptor disruption 🔴 |
| **Caffeine** | Low cellular toxicity — consistent with known safety profile ✅ |
| **Doxorubicin** | High SR-p53 — correctly identifies DNA damage mechanism 🔴 |
| **Metformin** | Low risk — consistent with excellent clinical safety record ✅ |

### Known Model Limitations
> Heroin, Benzene, and Thalidomide score **low** on our model — not because they are safe, but because their danger mechanisms fall **outside Tox21's scope:**
> - Heroin → opioid receptor binding (not a cellular toxicity mechanism)
> - Benzene → chronic bone marrow toxicity over years (not acute cellular)
> - Thalidomide → teratogenicity in embryos (no cell-based assay captures this)
>
> This reflects scientific maturity, not a model flaw. A complete safety system requires multiple complementary assay panels.

---

## 🚀 Differentiators vs Other Teams

| Feature | Typical Team | ToxPredict |
|---|---|---|
| Data split | Random split (leaky) | **Scaffold split (zero leakage)** |
| Feature count | 9 descriptors | **1033+ (Morgan FP + toxicophores)** |
| Model type | Single XGBoost | **Ensemble of 3 models** |
| Drug input | SMILES only | **Drug name OR SMILES** |
| Molecule view | 2D image | **Interactive 3D rotation** |
| Explainability | Basic importance | **SHAP + toxicophore alerts** |
| Drug safety | Toxicity only | **Full ADMET profile** |
| App modes | Single drug | **Single + Compare + Batch** |
| Validation | Test accuracy | **Real FDA drug validation** |

---

## 🔮 Limitations & Future Work

### Current Limitations
- Tox21 covers only 12 specific cellular assay types
- Does not predict addiction, neurotoxicity, or chronic exposure effects
- Morgan fingerprints miss 3D conformational information
- Model trained on ~7,800 compounds — larger datasets would improve generalization

### Future Enhancements
- [ ] Graph Neural Networks (GNNs) for molecular graph-based learning
- [ ] 3D shape descriptors using RDKit conformer generation
- [ ] Integration with ChEMBL for larger training data
- [ ] Multi-task learning across all 12 targets simultaneously
- [ ] Deployment on Streamlit Cloud for public access
- [ ] Support for protein-ligand docking scores

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
streamlit
rdkit
plotly
requests
py3Dmol
jupyter
ipykernel
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Team

**Dhruv Bajpai**
**Samarth Shukla**
**Kshitij Trivedi**
- B.Tech CSE | VIT Bhopal


---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- **Tox21 Dataset** — National Toxicology Program & NIH
- **RDKit** — Open-source cheminformatics library
- **PubChem** — Free chemical structure database (NIH)
- **3Dmol.js** — 3D molecular visualization library
- **SHAP** — Lundberg & Lee, 2017

---

<div align="center">

**Built for CodeCure Biohackathon | IIT BHU | Track A**

*Predicting drug toxicity with AI — making drug discovery safer and faster.*

</div>
