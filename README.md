# ECE9611/9612 – Residential Electricity Consumption Forecasting

Short-term (6-hour ahead) residential electricity consumption forecasting using
Sri Lankan smart meter data and household survey information.

This repository contains the code and notebooks used for the course project.

---

## Folder Structure

- **01 Dataset/**  
  Raw and intermediate data used for modelling.
  - `01 Raw Data from IEEE DataPort/` – original survey and smart meter CSVs.  
  - `02 Processed Data/` – outputs from the preprocessing scripts.  
  - `03 Data for model/` – final train/test CSVs and feature documentation
    (`Train_set.csv`, `Test_set.csv`, `README.md`).

- **02 Preprocessing/**  
  Python scripts to build the modelling dataset from raw files:
  - `01 Merge_survey.py` – merge household surveys data. This code merges eight surveys completed by users, based on household ID.
  - `02 Select_2024_Jan_w1_users.py` – Users who completed all 8 surveys in January 2024 are considered valid users.
  - `03 Smart_6hour_merge_from_Jan2023_to_Jan2024.py` – The meter reading data is extensive and originally stored in six separate tables, which need to be merged.
  - `04 Summary_survey_and_meter_data.py` – join smart meter and survey features.
  - `05 Add_label.py` – create the `future_6h_consumption` Calculate the labels. Based on the recorded readings, compute the difference between the reading at each time point and the reading 6 hours later as the label.

- **03 Feature Engineering/**  
  - `01 Feature_viewing_management.ipynb` – EDA and feature engineering
    (time features, historical consumption statistics, survey-based features, etc.).

- **04 Models/**  
There are 3 folders in total. Each folder allows you to view the results for each type of model.
  - Linear
  - Tree
  - ANN

- **05 Results Analysis/**
  - `Comparison of MAE, RMSE, and R².ipynb` – compare model performance.
  - `Feature importance analysis.ipynb` – analyze feature importance for tree models.
  - `Test Error Analysis Across Models.ipynb` – analyze prediction errors.
  - `True vs. Predicted.ipynb` – visualize true vs. predicted consumption values.



