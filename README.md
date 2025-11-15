# ❤️ Heart Disease Prediction Using Machine Learning  
A simple and beginner-friendly Machine Learning project that predicts the risk of heart disease based on user-provided medical parameters.  
The model uses **Logistic Regression**, and the system runs entirely on Python.

## 📌 Project Overview
This project takes essential health inputs such as age, blood pressure, cholesterol level, chest pain type, ECG readings, and physical exercise-related symptoms.  
These values are converted into numerical format, scaled, and passed into a trained ML model which predicts the probability of heart disease.

## 🚀 Features
- Logistic Regression based prediction  
- Clean and interpretable medical ML system  
- User input → processing → prediction pipeline  
- Fully beginner-friendly  
- Includes model, scaler, and dataset  
- Gives risk % and suggestions  

## 🧠 How It Works (Workflow)

### 1️⃣ User Input  
The user enters medical values like:
- Age  
- Sex  
- Chest Pain Type  
- Resting BP  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Max Heart Rate  
- Exercise Angina  
- Oldpeak  
- ST Slope  

### 2️⃣ Data Preprocessing  
- Text → Numeric conversion  
- Scaling using StandardScaler  
- Matching same feature order as training  

### 3️⃣ Model Prediction  
The trained Logistic Regression model (`best_model.pkl`) predicts:
- **0 → No Heart Disease**
- **1 → Heart Disease Likely**

### 4️⃣ Output  
Displays:
- Final prediction  
- Probability score  
- Possible medical reasons  
- Health suggestions  

## 📂 Project Structure

📁 heart-disease-prediction/
│── heart_pred_human.py # Main prediction script
│── best_model.pkl # Trained logistic regression model
│── scaler.pkl # StandardScaler for preprocessing
│── heart.csv # Dataset used
│── README.md # Project documentation

## 📊 Dataset Information
Dataset Used: **Cleveland Heart Disease Dataset**  
- Rows: 303  
- Features: 14  
- Target:  
  - **1 = Disease Present**  
  - **0 = No Disease**

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Logistic Regression  
- StandardScaler  
- Pickle  

## ▶️ How to Run

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/heart-disease-prediction.git

2. Install dependencies
pip install -r requirements.txt

3. Run the predictor
python heart_pred_human.py

👨‍⚕️ Example Input
Enter Age: 20
Enter Sex (Male/Female): Male
Enter Chest Pain Type (ATA, NAP, ASY, TA): TA
Enter Resting Blood Pressure (mm Hg): 160
Enter Cholesterol (mg/dl): 240
Fasting Blood Sugar > 120 mg/dl? (1 for Yes, 0 for No): 1
Enter Resting ECG (Normal, LVH, ST): ST
Enter Maximum Heart Rate Achieved: 180
Exercise Induced Angina? (Y/N): Y
Enter Oldpeak (ST Depression): 2
Enter ST Slope (Up, Flat, Down): Up

🏁 Conclusion

This project demonstrates how machine learning can be applied to medical risk prediction.
It is beginner-friendly, easy to understand, and can be expanded with advanced models like Random Forest, SVM, or XGBoost.

👤 Author

MD HABIB ALAM
ID: 231003003041
Email: mdhabib231003003041@technoindiaeducation.com
