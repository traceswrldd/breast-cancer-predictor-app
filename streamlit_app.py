import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("🩺 Breast cancer predictor")

df = pd.read_csv('wisconsin_breast_data.csv')

df = df.drop(['id', 'Unnamed: 32'], axis=1)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

st.write("Processed DataFrame (first 5 rows):")
st.dataframe(df.head())
st.write(f"Shape of features (X): {X.shape}")
st.write(f"Shape of target (y): {y.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

st.write("StandardScaler and LogisticRegression model trained successfully.")

st.title("Breast Cancer Prediction App")
st.write("This app predicts the probability of a breast mass being malignant based on various features.")

st.sidebar.header("Enter Patient Features")

feature_means = X.mean()

user_input = {}
for feature in X.columns:
    user_input[feature] = st.sidebar.number_input(
        f"{feature}:",
        min_value=float(X[feature].min()),
        max_value=float(X[feature].max()),
        value=float(feature_means[feature]),
        step=0.01, # Added step for finer control
        format="%.4f" # Added format for better display of floats
        )

# Display the collected user input (optional, for verification)
# st.write("User Input:")
# st.write(user_input)

def predict_malignancy_probability_app(user_input):
  """
  Predicts the probability of malignancy for a new patient based on user input.

  Args:
    user_input: A dictionary containing the feature values for the new patient.

  Returns:
    The predicted probability of malignancy.
  """

user_input_df = pd.DataFrame([user_input], columns=X.columns)
  new_patient_data = user_input_df.values.reshape(1, -1)

new_patient_scaled = scaler.transform(new_patient_data)

probability = model.predict_proba(new_patient_scaled)[0][1]

  return probability

if st.sidebar.button("Predict"):
    # Get the prediction
    probability = predict_malignancy_probability_app(user_input)

st.subheader("Prediction Result")
    st.write(f"The probability of malignancy is: {probability:.4f}")

    if probability > threshold:
        st.write("Based on the prediction, the mass is likely malignant.")
    else:
        st.write("Based on the prediction, the mass is likely benign.")

