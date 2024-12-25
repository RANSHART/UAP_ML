import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Function to load the trained model
def load_model():
    return joblib.load("uap_model.pkl")

def load_feature_names():
    return joblib.load("feature.pkl")

# Load model and feature names
model = load_model()
feature_names = load_feature_names()

# Streamlit app setup
st.title("App Klasifikasi Tingkat Polusi Udara (UAP)")

# Sidebar for user input
st.sidebar.header("Input Features")

# Collect user input for all features
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.slider(
        f"Enter {feature}", 
        min_value=0.0, 
        max_value=10000.0,  # As an example, assuming CO and other features have a range of 0-10
        value=1.0, 
        step=0.1
    )

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display prediction result
        st.write("### Prediction Result")
        if prediction[0] == 0:
            st.write("Pollution Level: Low - **Very Good**")
        elif prediction[0] == 1:
            st.write("Pollution Level: Medium - **Play Safe**")
        else:
            st.write("Pollution Level: High - **Danger**")
        st.write(f"Probability of High Pollution: {prediction_proba[0][2]:.2f}")

        # Feature importance visualization
        if hasattr(model, "feature_importances_"):  # Check if feature importance is available
            feature_importances = model.feature_importances_
            y_pos = np.arange(len(feature_names))

            # Plot feature importance
            st.write("### Feature Importance")
            plt.figure(figsize=(8, 6))
            plt.barh(y_pos, feature_importances, align='center')
            plt.yticks(y_pos, feature_names)
            plt.xlabel("Importance")
            plt.title("Feature Importance from Model")
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.warning("Feature importance is not available for this model.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by [182-Rangga Saputra Hari Pratama]")
