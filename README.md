# PDX Drug Sensitivity Prediction

A machine learning framework for predicting drug sensitivity in Patient-Derived Xenograft (PDX) models of T-cell Acute Lymphoblastic Leukemia (T-ALL) using morphological and protein expression features.

## Overview

This repository contains the implementation of a machine learning approach to predict sensitivity to targeted therapies (Dasatinib and Venetoclax) in T-ALL PDX models. The approach uses XGBoost classification models trained on cell morphology parameters and protein expression data with a focus on drug-specific mechanisms (LCK pathway for Dasatinib and BCL2 pathway for Venetoclax).

## Key Features

- **Drug-specific feature engineering** tailored to the mechanism of action of each drug
- **PDX-focused rotational cross-validation** that maintains PDX group integrity
- **Comprehensive feature importance analysis** using SHAP (SHapley Additive exPlanations)
- **Visualization tools** for model performance, feature importance, and UMAP embeddings
- **Optimized hyperparameters** for XGBoost models specific to each drug

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- UMAP
- Imbalanced-learn (for SMOTE)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pdx-drug-sensitivity.git
cd pdx-drug-sensitivity

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main workflow is implemented in `pdx_drug_sensitivity.py`:

```python
# Run the complete analysis pipeline
python pdx_drug_sensitivity.py
```

To use the model for prediction on new data:

```python
from pdx_drug_sensitivity import train_and_evaluate_xgboost

# Train model with your data
result = train_and_evaluate_xgboost(
    X_train, y_train, X_val, y_val, X_test, y_test, 
    feature_names, drug_name="dasatinib", k_best=30
)

# Access the trained model
model = result['model']
```

## Methodology

### Feature Engineering

The script implements various feature engineering techniques:
- Basic protein expression features (pLCK, LCK, pBCL2, BCL2)
- Ratio features (pLCK/LCK, pBCL2/BCL2)
- Log transformations and log ratios
- Z-score normalization within PDX groups
- Interaction features between protein expressions and cell morphology

### Cross-Validation Strategy

A PDX group-aware rotational cross-validation approach is used to maintain the biological integrity of the models:
- Samples from the same PDX group are kept together in train/validation/test splits
- Multiple rotations of PDX groups ensure robust model evaluation
- Separate optimal rotations are identified for each drug

### Model Training

XGBoost classification models with drug-specific hyperparameters:
- Feature selection using SelectKBest with ANOVA F-value
- SMOTE for handling class imbalance
- Early stopping using validation set
- Hyperparameter tuning specific to each drug

### Visualization and Analysis

- ROC curves and confusion matrices for model performance
- Probability distribution plots
- SHAP analysis for feature importance
- UMAP visualization for sample clustering based on top features

## Structure

```
.
├── pdx_drug_sensitivity.py  # Main script with all functionality
├── requirements.txt         # Required packages
├── data/                    # Data directory (not included)
└── outputs/                 # Generated outputs
    ├── models/              # Saved model files
    ├── plots/               # Generated visualizations
    └── data/                # Output data files
```

## Helper Functions

The repository includes numerous helper functions for:
- Data preprocessing and feature engineering
- Model training and evaluation
- Visualization of results
- SHAP analysis of feature importance
- UMAP dimensionality reduction

## Example Output

The analysis generates multiple visualizations and data files:
- Performance metrics for each rotation
- ROC curves and confusion matrices
- Probability distribution plots
- SHAP summary plots
- Feature importance rankings
- UMAP visualizations
- Comprehensive PDF reports

## Citation

If you use this code in your research, please cite:

```
@article{xx,
  title={μPharma: A Microfluidic AI-driven Pharmacotyping Platform for Single-cell Drug Sensitivity Prediction in Leukemia},
  author={Huiqian Hu, Alphonsus H. C. Ng, Yue Lu},
  journal={xx},
  year={2025},
  volume={xx},
  pages={xx}
}
```

## License

[MIT License](LICENSE)

## Acknowledgements

Thanks to our collaborators Dr. Jun J Yang and Dr. Huanbin Zhao from St. Jude Children’s Research Hospital who contributed to this project and provided the PDX data.

## Contact

For questions or feedback, please contact [digipharma21@gmail.com].
