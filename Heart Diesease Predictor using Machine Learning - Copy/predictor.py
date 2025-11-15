import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and encoder
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Try loading encoder if used during training
try:
    with open("encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
except FileNotFoundError:
    encoder = None

print("\nAI Heart Disease Prediction System\n")

# Take user input
age = int(input("Enter Age: "))
sex = input("Enter Sex (Male/Female): ").capitalize()
chest_pain = input("Enter Chest Pain Type (ATA, NAP, ASY, TA): ").upper()
rest_bp = int(input("Enter Resting Blood Pressure (mm Hg): "))
chol = int(input("Enter Cholesterol (mg/dl): "))
fast_bs = int(input("Fasting Blood Sugar > 120 mg/dl? (1 for Yes, 0 for No): "))
rest_ecg = input("Enter Resting ECG (Normal, LVH, ST): ").capitalize()
max_hr = int(input("Enter Maximum Heart Rate Achieved: "))
ex_ang = input("Exercise Induced Angina? (Y/N): ").upper()
oldpeak = float(input("Enter Oldpeak (ST Depression): "))
st_slope = input("Enter ST Slope (Up, Flat, Down): ").capitalize()

# Prepare input
patient_data = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain,
    'RestingBP': rest_bp,
    'Cholesterol': chol,
    'FastingBS': fast_bs,
    'RestingECG': rest_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': ex_ang,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope
}

input_df = pd.DataFrame([patient_data])

# Apply encoder if it exists
if encoder is not None:
    cat_cols = encoder.feature_names_in_
    encoded_data = pd.DataFrame(
        encoder.transform(input_df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )
    input_df = input_df.drop(columns=cat_cols)
    input_df = pd.concat([input_df, encoded_data], axis=1)

# Match column order for the model
X_columns = scaler.feature_names_in_
input_df = input_df.reindex(columns=X_columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

if prediction == 1:
    print(f"The patient is likely to have Heart Disease (Risk Probability: {probability*100:.2f}%)")

    print("\nPossible Reasons:")
    if rest_bp > 140:
        print("- High blood pressure detected.")
    if chol > 240:
        print("- High cholesterol level.")
    if oldpeak > 2:
        print("- Abnormal ST depression.")
    if ex_ang == "Y":
        print("- Chest pain during exercise (possible angina).")

    print("\nRecommended Steps:")
    print("- Consult a cardiologist immediately.")
    print("- Maintain a low-fat, low-salt diet.")
    print("- Take prescribed medicines as advised.")
    print("- Regular morning walks and avoid stress.")
    print("- Avoid smoking and alcohol.\n")

else:
    print(f"The patient is healthy (Heart Disease Probability: {probability*100:.2f}%)")

    print("\nTips to stay healthy:")
    print("- Maintain a balanced diet and regular exercise.")
    print("- Keep stress levels low.")
    print("- Monitor BP and cholesterol regularly.\n")
