# ECE9611/9612 – Residential Electricity Consumption Forecasting

Short-term (6-hour ahead) residential electricity consumption forecasting using
Sri Lankan smart meter data and household survey information.

This repository contains the code and notebooks used for the course project.

---

## Folder Structure

- **01 Dataset/**  
  Raw and intermediate data (6-hour smart meter readings and survey CSVs).

- **02 Preprocessing/**  
  Python scripts to build the modelling dataset:
  - `01 Merge_survey.py` – build household-level survey features.
  - `02 Select_2024_Jan_w1_users.py` – select valid Wave 1 households (Jan 2024).
  - `03 Smart_6hour_merge_from_Jan2023_to_Jan2024.py` – merge 6-hour meter files.
  - `04 Summary_survey_and_meter_data.py` – join smart meter and survey features.
  - `05 Add_label.py` – create `future_6h_consumption` label.

- **03 Feature Engineering/**  
  - `01 Feature_viewing_management.ipynb` – EDA and additional feature engineering
    (time features, lagged consumption, etc.).

- **04 Models/**  
  - `01 model.ipynb` – training and evaluation of ML models
    (Linear Regression, Random Forest, XGBoost, LightGBM, etc.).

---


## Main Result (LightGBM)

Using the feature-engineered dataset and a household-based train/test split,
LightGBM achieves approximately:

- **MAE:** ~0.39 kWh
- **R²:** ~0.61

