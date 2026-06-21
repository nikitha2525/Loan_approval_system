# 🏦 Loan Approval Prediction System

A machine learning web application that predicts loan approval status using a Random Forest model, built with Flask and deployed on Vercel.

## 🚀 Live Demo
[loan-approval-system-puce.vercel.app](https://loan-approval-system-puce.vercel.app)

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **ML Model:** Random Forest Classifier (scikit-learn)
- **Frontend:** HTML, CSS (custom dark theme)
- **Deployment:** Vercel

## 📁 Project Structure
```
loan_approval_system/
│
├── app.py                  # Flask application
├── vercel.json             # Vercel deployment config
├── pyproject.toml          # Python project config
├── requirements.txt        # Dependencies
├── .gitignore
├── README.md
│
├── model.pkl               # Trained Random Forest model
├── le_gender.pkl           # Gender label encoder
├── le_married.pkl          # Marital status label encoder
├── le_prop.pkl             # Property area label encoder
├── thresholds.json         # Income/loan quantile thresholds
│
└── templates/
    ├── loan.html           # Home page
    └── approval.html       # Prediction form
```

## ⚙️ Features
- Predicts loan approval based on applicant details
- Income threshold business rule enforcement
- Probability score shown with result
- Fully responsive dark UI
- No database required

## 🧠 Model Details
- Algorithm: Random Forest Classifier
- Target: `Loan_Status_Rejected` (1 = Rejected, 0 = Approved)
- Features: Gender, Age, Income, Loan Amount, Credit History, etc.

## 🏃 Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Visit: `http://localhost:5000`
