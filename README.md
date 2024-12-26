# Air Quality Classification Project

## Overview
This project involves analyzing and classifying air quality based on CO (Carbon Monoxide) concentrations measured in the air. The dataset, `AirQuality.csv`, contains several air quality metrics, and we focus on predicting the level of air pollution using classification algorithms.

## Objectives
- Preprocess the air quality dataset to make it suitable for machine learning models.
- Categorize air pollution levels into three classes: `Low`, `Medium`, and `High`.
- Train and evaluate models using **Random Forest** and **XGBoost** classifiers.
- Save the best-performing model for future use.

## Dataset Description
The dataset includes various features related to air quality measurements. Key steps in preprocessing include:
- Dropping irrelevant columns like `Date`, `Time`, and unnamed columns.
- Converting the target column `CO(GT)` to numeric and handling non-numeric values.
- Categorizing CO levels into three classes based on percentiles.

### Target Classes:
- **Low (0)**: CO levels below or equal to the 50th percentile.
- **Medium (1)**: CO levels between the 50th and 75th percentiles.
- **High (2)**: CO levels above the 75th percentile.

## Methodology
1. **Data Preprocessing:**
   - Dropped unused columns.
   - Converted non-numeric values to numeric.
   - Handled missing data by removing rows with NaN values in `CO(GT)`.
   - Created a categorical target variable based on CO percentiles.

2. **Feature Engineering:**
   - Ensured all features were numeric, replacing NaN values with `0`.

3. **Model Training:**
   - Split the dataset into training and testing sets (80:20 split).
   - Trained two models: **Random Forest** and **XGBoost**.

4. **Evaluation:**
   - Evaluated models using metrics like accuracy and classification reports.
   - Generated learning curves to assess model performance across different training sizes.

5. **Model Deployment:**
   - Saved the best-performing model (`Random Forest`) using `joblib`.
   - Saved the feature names for future reference.

## Results
### Random Forest Classifier:
- **Classification Report:**
  Includes precision, recall, and F1-score for each class.
- **Accuracy:** Achieved on the test set.

### XGBoost Classifier:
- **Classification Report:**
  Includes precision, recall, and F1-score for each class.
- **Accuracy:** Achieved on the test set.

## Visualizations
- Learning curves for the Random Forest model demonstrate the model’s training and cross-validation performance across different training set sizes.

## How to Run the Project
1. **Dependencies:**
   Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib xgboost joblib
   ```

2. **Run the Script:**
   Place `AirQuality.csv` in the same directory as the script and execute the script.

3. **Output:**
   - The script prints evaluation metrics for both models.
   - Saves the best model as `uap_model.pkl`.
   - Saves feature names as `feature.pkl`.

4. **Plot Learning Curve:**
   The script generates a plot showing the learning curve of the Random Forest model.

## Files
- `AirQuality.csv`: Input dataset.
- `uap_model.pkl`: Saved Random Forest model.
- `feature.pkl`: Saved feature names.
- Script file: Python script containing the code.

## Future Improvements
- Experiment with hyperparameter tuning for both models to improve accuracy.
- Incorporate additional features or datasets for more comprehensive air quality predictions.
- Deploy the model as a web application for real-time predictions.

