import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load dataset
data = pd.read_csv("AirQuality.csv", delimiter=';')

# Drop unused columns, including 'Date' and 'Time'
data = data.drop(["Date", "Time", "Unnamed: 15", "Unnamed: 16"], axis=1, errors='ignore')

# Convert 'CO(GT)' to numeric and handle non-numeric values
data['CO(GT)'] = pd.to_numeric(data['CO(GT)'], errors='coerce')

# Drop rows with NaN in 'CO(GT)'
data = data.dropna(subset=['CO(GT)'])

# Calculate percentiles for CO(GT)
low_threshold = np.percentile(data['CO(GT)'], 50)  # 50th percentile
medium_threshold = np.percentile(data['CO(GT)'], 75)  # 75th percentile

# Multi-class classification based on percentiles
def categorize_pollution(co):
    if co <= low_threshold:
        return 0  # Low
    elif low_threshold < co <= medium_threshold:
        return 1  # Medium
    else:
        return 2  # High

data['CO(GT)'] = data['CO(GT)'].apply(categorize_pollution)

# Define features and target
X = data.drop("CO(GT)", axis=1)
y = data["CO(GT)"]

# Handle other non-numeric columns (convert to numeric or drop)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("XGBoost Report")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Save the best model (e.g., Random Forest)
joblib.dump(rf_model, "uap_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "feature.pkl")

# Generate and plot learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Plot learning curve for Random Forest
plot_learning_curve(rf_model, X_train, y_train)
