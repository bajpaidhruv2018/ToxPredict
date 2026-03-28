# 🧬 ToxPredict: AI-Powered Drug Toxicity Predictor

ToxPredict is an AI-powered machine learning application designed to predict the toxicity of molecules across 12 biological targets instantly. It uses structural and chemical properties of drugs, including basic molecular descriptors, Morgan fingerprints, and toxicophore detection to evaluate toxicity risks.

## 🚀 Features

- **Toxicity Prediction**: Predicts toxicity across 12 targets (e.g., NR-AR, SR-p53, etc.) using an ensemble model combining XGBoost, RandomForest, and Logistic Regression.
- **Automated Feature Extraction**: Computes basic descriptors (MolWt, LogP, TPSA, etc.), extracts Morgan fingerprints, and flags structural representations of 10 known toxicophores.
- **Drug Discovery Integration**: Try out predictions by effortlessly typing a drug name (fetches SMILES via PubChem/OPSIN/CIR) or by providing a custom SMILES string manually.
- **ADMET Profiling**: Provides a rapid ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) drug safety profile, evaluating properties like Oral Bioavailability, BBB Penetration, Hepatotoxicity Risk, and Lipinski's Rule of 5 compliance.
- **Interactive Web App**: Built with Streamlit for an intuitive UI, featuring graphical molecular rendering, dynamic risk scoring, and downloadable CSV target reports.

## 📂 Project Structure

```text
CodeCure/
│
├── app/
│   └── app.py                  # Main Streamlit web application
│
├── data/
│   ├── tox21.csv               # Raw Tox21 dataset
│   └── tox21_processed.csv     # Processed dataset with extracted features
│
├── notebooks/                  # Jupyter notebooks for interactive exploratory experimentation
│
├── results/                    # Generated metrics, plots, predictions and cached models
│   ├── class_balance.png       # Target class imbalance visualization
│   ├── metrics.csv             # Evaluation metrics for all the target models
│   └── models.pkl              # Pickled ensemble of trained model checkpoints
│
└── src/
    ├── 01_eda.py               # Exploratory Data Analysis script
    ├── 02_features.py          # Molecular feature extraction script
    ├── 03_train.py             # Model training and evaluation script
    └── 04_visualize.py         # Visualization scripts for model interpretation
```

## 🛠️ Installation & Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd CodeCure
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. You will need the following key libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit rdkit matplotlib seaborn requests
   ```

3. **Run the Data Pipeline** (Optional, if you wish to retrain the models from scratch):
   ```bash
   python src/01_eda.py
   python src/02_features.py
   python src/03_train.py
   ```

4. **Launch the Streamlit App**:
   ```bash
   streamlit run app/app.py
   ```

## 🧠 How It Works

1. **Input**: The user inputs a drug name or SMILES string.
2. **Feature Engineering**: The app uses `rdkit` to calculate molecular descriptors, Morgan fingerprints via `AllChem`, and detects structural toxicophores using SMARTS patterns.
3. **Prediction**: The generated feature vector is fed into pre-trained ensemble classifiers (`models.pkl`) to predict the probability of toxicity across 12 specific target endpoints from the Tox21 dataset.
4. **Insights**: Results are aggregated to produce an overall risk probability score and an ADMET profile, highlighting sub-structural threats and compliance with drug-likeness rules.

## 📊 Models Built

The core predictive engine is a `VotingClassifier` (Soft Voting) built combining the decisions of:
- **XGBClassifier**: Gradient boosting algorithm handling highly imbalanced target targets via proportional `scale_pos_weight`.
- **RandomForestClassifier**: Ensemble tree method using automatically adjusted balanced class weights.
- **LogisticRegression**: Scaled linear statistical model to calibrate output probabilities to the correct scale.
