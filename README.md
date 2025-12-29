# Prediction of Oncology Drug Responses from Gene Expression using Machine Learning

Bachelor's Thesis Project - Pan-drug Models on GDSC1 Data

## Project Overview

This project develops machine learning models to predict cancer drug responses (IC50 and AUC) from gene expression data using the GDSC1 (Genomics of Drug Sensitivity in Cancer) dataset.

### Objectives

1. Develop pan-drug models that predict drug responses across multiple drugs simultaneously
2. Compare three ML approaches: Random Forest, XGBoost, and Neural Networks (PyTorch)
3. Predict both IC50 and AUC metrics separately (6 models total)
4. Identify important genes that contribute to drug response prediction
5. Generate comprehensive thesis documentation in Romanian

### Key Features

- **Advanced implementations** with detailed code comments for learning
- **Drug embeddings** in neural networks (vs traditional one-hot encoding)
- **SHAP values** for model interpretability
- **TensorBoard** integration for training monitoring
- **Ensemble methods** combining multiple models
- **Publication-quality visualizations** (300 DPI, Romanian labels)

## Project Structure

```
D:\uni\licenta\
├── archive/                    # Raw GDSC1 dataset (original data)
├── data/
│   ├── processed/              # Cleaned and preprocessed data
│   └── splits/                 # Train/test/validation splits
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # ML model implementations
│   ├── training/               # Training logic and utilities
│   ├── evaluation/             # Metrics and evaluation
│   └── visualization/          # Plotting functions
├── scripts/                    # Executable scripts for main workflows
├── notebooks/                  # Jupyter notebooks for exploration
├── experiments/                # Model checkpoints and logs
├── results/
│   ├── metrics/                # Performance metrics (CSV/JSON)
│   └── figures/                # All thesis figures
├── chapters/                   # LaTeX chapter files (Romanian)
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── documentatie.tex            # Main LaTeX document
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster neural network training

### Installation

1. **Clone the repository** (or navigate to project directory)

```bash
cd D:\uni\licenta
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
```

3. **Activate the virtual environment**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Verify installation**

```bash
python -c "import torch; import xgboost; import pandas; print('All packages installed successfully!')"
```

## Usage

### 1. Data Preprocessing

Run the preprocessing pipeline to clean data, select features, and create train/test splits:

```bash
python scripts/01_preprocess_data.py
```

This will:
- Load GDSC1 dataset from `archive/`
- Handle missing values
- Filter drugs (keep drugs with ≥30 samples)
- Select top 5,000 genes by variance
- Normalize gene expression (Z-score)
- Create train/test splits (80/20 by cell line)
- Save processed data to `data/processed/`

### 2. Train Models

Train all 6 models (Random Forest, XGBoost, Neural Network × IC50, AUC):

```bash
python scripts/02_train_models.py
```

Options:
- `--model {rf,xgb,nn,all}` - Train specific model or all
- `--target {ic50,auc,both}` - Train for specific target or both
- `--cv` - Enable cross-validation for hyperparameter tuning
- `--gpu` - Use GPU for neural network training (if available)

Example:
```bash
python scripts/02_train_models.py --model all --target both --cv
```

### 3. Evaluate Models

Evaluate trained models on test set and generate metrics:

```bash
python scripts/03_evaluate_models.py
```

This generates:
- `results/metrics/model_comparison.csv` - Overall performance metrics
- `results/metrics/per_drug_performance.csv` - Performance breakdown by drug
- `results/metrics/feature_importance.csv` - Top important genes
- `results/predictions/*.csv` - Predicted vs actual values

### 4. Generate Figures

Create all thesis figures with Romanian labels:

```bash
python scripts/04_generate_figures.py
```

Output: 15+ publication-quality figures (300 DPI) in `results/figures/`

### 5. Compile LaTeX Documentation

```bash
pdflatex documentatie.tex
bibtex documentatie
pdflatex documentatie.tex
pdflatex documentatie.tex
```

## Model Descriptions

### Random Forest
- **Purpose**: Baseline model + feature importance analysis
- **Hyperparameters**: 500 estimators, max_depth=20
- **Advantage**: Robust, extracts gene importance scores

### XGBoost
- **Purpose**: Strong baseline, typically best performer on tabular data
- **Hyperparameters**: learning_rate=0.05, max_depth=6, n_estimators=1000
- **Advantage**: State-of-the-art gradient boosting, GPU acceleration

### Neural Network (PyTorch)
- **Architecture**: [5000 genes + 64-dim drug embedding] → 1024 → 512 → 256 → 128 → 1
- **Features**: Batch normalization, dropout (0.2-0.3), early stopping
- **Advantage**: Can learn complex non-linear patterns, drug embeddings

## Expected Performance

Based on literature, reasonable performance targets for this task:

- **R² (AUC prediction)**: 0.35 - 0.60 (AUC is more stable)
- **R² (IC50 prediction)**: 0.25 - 0.50 (IC50 is noisier)
- **Spearman correlation**: 0.50 - 0.70

**Note**: Drug response prediction is inherently noisy due to biological variability. R² values in this range are considered successful for a bachelor's thesis.

## Dataset Information

**Source**: GDSC1 (Genomics of Drug Sensitivity in Cancer)

**Location**: `D:\uni\licenta\archive\GDSC_DATASET.csv`

**Contents**:
- 17,419 gene expression features per sample
- Multiple drugs tested across cancer cell lines
- Response metrics: IC50 (concentration for 50% inhibition), AUC (area under dose-response curve)
- Cell line metadata: tissue type, cancer type

**Citation**: Yang, W. et al. (2013). "Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells." Nucleic Acids Research.

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Random seed: 42
- Set in: `src/config.py`

To reproduce results:
1. Use the same data splits (saved in `data/splits/`)
2. Use the same hyperparameters (documented in `experiments/*/config.yaml`)
3. Run scripts in order: preprocess → train → evaluate → visualize

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
```

### Linting

```bash
flake8 src/ scripts/ tests/
```

## Jupyter Notebooks

Explore data and results interactively:

1. `notebooks/01_data_exploration.ipynb` - EDA and data analysis
2. `notebooks/02_preprocessing_validation.ipynb` - Validate preprocessing
3. `notebooks/03_model_experiments.ipynb` - Quick model experiments
4. `notebooks/04_results_generation.ipynb` - Generate additional results

## Thesis Chapters (Romanian)

1. **Introducere** (`chapters/01_introduction.tex`) - Introduction and objectives
2. **Date și Preprocesare** (`chapters/02_data.tex`) - Data description and preprocessing
3. **Metode** (`chapters/03_methods.tex`) - ML methods and architectures
4. **Setup Experimental** (`chapters/04_experiments.tex`) - Experimental setup
5. **Rezultate** (`chapters/05_results.tex`) - Results and analysis
6. **Discuții și Limitări** (`chapters/06_discussion.tex`) - Discussion and limitations

## Troubleshooting

### Out of Memory (OOM) errors
- Reduce batch size in `src/config.py` (default: 128)
- Reduce number of genes (default: 5000)
- Use CPU instead of GPU for neural networks

### Neural network not converging
- Check learning rate (try reducing to 0.0001)
- Increase number of epochs (max: 200)
- Check for data leakage in train/test split

### Poor model performance (R² < 0.2)
- Verify preprocessing is correct
- Check for bugs in data loading
- Compare to published baselines in literature
- Consider increasing number of genes or using different feature selection

## Contact

Bachelor's Thesis Project
2026

## License

Academic use only - Bachelor's Thesis Project
