# PDX Drug Sensitivity Prediction Demo

## μPharma Platform Demonstration

This repository contains a demonstration notebook for our PDX-specific drug sensitivity prediction methodology, as described in our manuscript.

## Quick Start

### Option 1: Google Colab (Recommended)
1. Upload `demo_pdx_drug_sensitivity.ipynb` to Google Colab
2. Run all cells sequentially
3. No installation required - the notebook installs dependencies automatically

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap umap-learn imbalanced-learn

# Launch Jupyter
jupyter notebook demo_pdx_drug_sensitivity.ipynb
```

## What This Demo Shows

The notebook demonstrates our complete methodology using synthetic data:

### 1. **Data Structure**
- 14 PDX groups matching our experimental design
- Morphological features (AreaShape metrics)
- Protein expression features (LCK and BCL2 pathways)

### 2. **Feature Engineering**
- Drug-specific ratio features
- Log transformations
- Within-PDX group normalization
- Protein-morphology interaction features

### 3. **PDX Rotational Cross-Validation**
- Predefined rotations ensuring balanced train/val/test splits
- PDX group integrity maintained (no data leakage)
- Drug-specific rotation schemes

### 4. **XGBoost Models**
- Drug-specific hyperparameters
- Feature selection (SelectKBest)
- SMOTE for class imbalance

### 5. **Interpretability**
- SHAP analysis for feature importance
- UMAP visualization from SHAP values
- Feature consistency across rotations

## Key Results to Observe

When you run the demo, pay attention to:

1. **Feature Selection**: Notice how Dasatinib models select LCK features while Venetoclax models select BCL2 features
2. **Performance Metrics**: ROC AUC typically ranges 0.7-0.9 with synthetic data
3. **SHAP Analysis**: Top features align with drug mechanisms
4. **UMAP Clustering**: Clear separation of sensitive/resistant samples

## Understanding the Output

### Cross-Validation Results
- Each rotation represents a different train/val/test split
- Performance varies across rotations
- Average metrics provide robust estimates

### SHAP Plots
- **Bar plots**: Overall feature importance
- **Summary plots**: Feature impact on predictions
- **UMAP from SHAP**: Sample clustering based on feature contributions

### Feature Consistency
The demo shows which features are consistently selected across rotations, indicating robust predictive markers.

## Comparison to Full Pipeline

| Aspect | Demo (Synthetic) | Full Pipeline (Real Data) |
|--------|-----------------|-------------------------|
| Samples | ~2,100 synthetic | ~15,000+ real |
| PDX Groups | 14 | 14 |
| Rotations | 2 per drug | 4+ per drug |
| Features | ~30 | 100+ |
| Runtime | ~2 minutes | ~30 minutes |
| Performance | 0.7-0.9 AUC | 0.85-0.95 AUC |

## Customization

To use your own data:

1. Replace the `generate_synthetic_pdx_data()` function with your data loader
2. Ensure your data has columns:
   - `Group`: PDX group identifier
   - `AreaShape_*`: Morphological features
   - `Intensity_*_LCK/pLCK`: LCK pathway features
   - `Intensity_*_BCL_2/pBCL2`: BCL2 pathway features
   - `Dasatinib Sensitivity`: Binary (0/1)
   - `Venetoclax Sensitivity`: Binary (0/1)

3. Adjust PDX rotations if using different groups

## Citation

If you use this code, please cite:

```bibtex
@article{upharma2025,
  title={μPharma: A Microfluidic AI-driven Pharmacotyping Platform for 
         Single-cell Drug Sensitivity Prediction in Leukemia},
  author={Hu, Huiqian and Ng, Alphonsus H. C. and Lu, Yue},
  journal={[Journal Name]},
  year={2025}
}
```

## Contact

For questions: digipharma21@gmail.com

## License

This demo code is provided for academic use under MIT License.
