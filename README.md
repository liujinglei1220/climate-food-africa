
# Research on the Impact of Climate Change on Food Security in Africa

This repository contains Python code used in the study titled "Research on the Impact of Climate Change on Food Security in Africa". The scripts support data analysis, model training, prediction, and uncertainty estimation for four major crops.

## File Descriptions

- `RF_best.py`  
  Trains optimized Random Forest (RF) models for crop yield prediction.

- `RF_best_predict.py`  
  Uses the trained RF models to project future crop yields under CMIP6 climate scenarios.

- `RF_predict.py`  
  Performs historical crop yield prediction using the trained models.

- `Timeserise.py`  
  Implements temporal cross-validation, including GroupKFold and Leave-One-Year-Out strategies.

- `crop_density.py`  
  Produces scatter plots illustrating model accuracy for each of the four crop types.

- `Importance.py`  
  Calculates and visualizes feature importance from the trained RF models.

- `Sigma_decade.py`  
  Computes decadal uncertainty by combining inter-model (GCM) variability and model-related variance.

## Requirements

The scripts are written in Python 3 and require the following packages:

- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- joblib

See `requirements.txt` for installation details.

## Data Availability

All datasets used in this study are publicly available and are described in detail in the manuscript.

## Citation

Liu et al. (under review). Research on the Impact of Climate Change on Food Security in Africa. *Scientific Reports*. DOI: To be added upon acceptance.
