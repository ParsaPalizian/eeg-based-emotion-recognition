# EEG-Based Emotion Recognition (VR Video Stimuli)

## 🎯 Goal
Predict a subject’s **emotional state** (*happy / sad / neutral*, or binary labels in some stages) from EEG recordings collected while watching affective videos, using a full pipeline of **feature extraction → feature selection → classification**.

## 🧪 Methods & Pipeline

### 1) Feature Extraction
- Extracted time-domain and frequency-domain EEG features using **NumPy**, **SciPy**, and **statsmodels**.
- Applied preprocessing (normalization, artifact handling) and computed descriptive features such as Hjorth parameters and cross-correlations.
- Generated consolidated feature tables saved as CSV for downstream analysis.

### 2) Feature Selection with PSO
- Implemented **Particle Swarm Optimization (PSO)** to select the most informative features.
- Balanced accuracy vs. feature count for compact model representation.
- Produced CSV summaries: `features_count.csv`, `fishersOfFeatures.csv`, and `selected_features.csv`.

### 3) Classification
- Used two neural architectures:
  - **MLP (Multi-Layer Perceptron)** — simple feedforward neural net for tabular features.
  - **RBF Network (Radial Basis Function)** — custom `RBFLayer` with adaptive centers and gamma parameters.
- Evaluated models using accuracy and confusion matrices across K-fold splits.

## 📊 Results
| Model | Accuracy | Macro-F1 | Notes |
|--------|-----------|-----------|--------|
| MLP | *(to fill)* | *(to fill)* | Hidden layers, activations |
| RBF | *(to fill)* | *(to fill)* | Gamma, centers |

📂 Repository Structure
```text
eeg-based-emotion-recognition/
├── src/           # Notebooks & scripts
│   ├── Hosh_Project_feature_extraction.ipynb
│   ├── Hosh_Project_PSO.ipynb
│   └── Hosh_Project_mlp_rbf.ipynb
├── data/          # CSVs / selected feature tables (no raw EEG)
│   ├── features_count.csv
│   ├── fishersOfFeatures.csv
│   └── selected_features.csv
├── report/        # Project report (PDF)
│   └── Hosh_Project_Report.pdf
├── .gitignore
└── README.md
```

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn scipy matplotlib statsmodels tensorflow
  ```
   
2. Run notebooks in order:

Hosh_Project_feature_extraction.ipynb

Hosh_Project_PSO.ipynb

Hosh_Project_mlp_rbf.ipynb

3. Place CSVs in /data and report results.

## 🧠 Tools
Python • NumPy • pandas • scikit-learn • TensorFlow/Keras • SciPy • statsmodels • matplotlib

## 📄 Report
Full methodology and analysis are described in `report/Hosh_Project_Report.pdf`.

## 👤 Author
**Parsa Palizian**  
Sharif University of Technology  
[GitHub Profile](https://github.com/ParsaPalizian)
