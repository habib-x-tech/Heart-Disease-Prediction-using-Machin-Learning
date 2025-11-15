import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
print("Loading dataset...")
df = pd.read_csv("heart.csv")

# Detect target column
target = "HeartDisease"
print(f"Target column detected: {target}")

# Encode categorical columns
print("Encoding categorical columns...")
df_encoded = pd.get_dummies(df, drop_first=True)

# Handle missing values
if df_encoded.isnull().sum().sum() > 0:
    print("Missing values detected! Filling with mean/mode...")
    for col in df_encoded.columns:
        if df_encoded[col].dtype in ['float64', 'int64']:
            df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)
        else:
            df_encoded[col].fillna(df_encoded[col].mode()[0], inplace=True)
else:
    print("No missing values found.")

# Split features and target
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# Balance dataset using SMOTE
print("Balancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model trained successfully and saved as best_model.pkl & scaler.pkl")
