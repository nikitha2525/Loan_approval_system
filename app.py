from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load model and encoders once at startup ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model      = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"),      "rb"))
le_gender  = pickle.load(open(os.path.join(BASE_DIR, "le_gender.pkl"),  "rb"))
le_married = pickle.load(open(os.path.join(BASE_DIR, "le_married.pkl"), "rb"))
le_prop    = pickle.load(open(os.path.join(BASE_DIR, "le_prop.pkl"),    "rb"))

with open(os.path.join(BASE_DIR, "thresholds.json")) as f:
    thresholds = json.load(f)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
@app.route('/loan')
def loan():
    return render_template('loan.html')


@app.route('/approval', methods=['GET', 'POST'])
def approval():
    return render_template('approval.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── Collect form inputs ───────────────────────────────────────────────
        Name               = request.form.get('name', '')
        age                = float(request.form.get('age', 0))
        gender             = request.form.get('gender', '')
        Martial_status     = request.form.get('Martial_status', '')
        education          = request.form.get('education', '')
        annual_income      = float(request.form.get('annual_income', 0))
        Employment_type    = request.form.get('Employment_type', '')
        years_exp          = float(request.form.get('years_of_experience', 0))
        existing_loan      = float(request.form.get('Existing_loan', 0))
        loan_amount        = float(request.form.get('Loan_amount', 0))
        loan_term          = float(request.form.get('Loan_term', 0))
        credit_history     = float(request.form.get('Credit_history', 0))
        dependents         = request.form.get('Dependents', '0')
        coapplicant_income = float(request.form.get('Coapplicant_Income', 0))
        property_area      = request.form.get('Property_Area', 'Urban')
        employment_status  = request.form.get('Employment_Status', 'Salaried')

        # ── Encode categorical inputs ─────────────────────────────────────────
        try:
            gender_enc = int(le_gender.transform([str(gender)])[0])
        except Exception:
            gender_enc = 0

        try:
            married_enc = int(le_married.transform([str(Martial_status)])[0])
        except Exception:
            married_enc = 0

        try:
            prop_enc = int(le_prop.transform([str(property_area)])[0])
        except Exception:
            prop_enc = 0

        # ── Build feature row matching model's training columns ───────────────
        cols = list(model.feature_names_in_)
        row  = {}

        for c in cols:
            if   c == 'Gender':                          row[c] = gender_enc
            elif c == 'Married':                         row[c] = married_enc
            elif c == 'Dependents':
                try:    row[c] = int(dependents)
                except: row[c] = 0
            elif c == 'Applicant_Income':                row[c] = annual_income
            elif c == 'Coapplicant_Income':              row[c] = coapplicant_income
            elif c == 'Loan_Amount':                     row[c] = loan_amount
            elif c == 'Loan_Term':                       row[c] = loan_term
            elif c == 'Credit_History':                  row[c] = credit_history
            elif c == 'Property_Area':                   row[c] = prop_enc
            elif c == 'Age':                             row[c] = age
            elif c == 'Employment_Status_Self-Employed': row[c] = 1 if employment_status == 'Self-Employed' else 0
            elif c == 'Employment_Status_Unemployed':    row[c] = 1 if employment_status == 'Unemployed'    else 0
            else:                                        row[c] = 0

        feature = pd.DataFrame([row], columns=cols)

        # ── Predict ───────────────────────────────────────────────────────────
        if hasattr(model, 'predict_proba'):
            probs          = model.predict_proba(feature)
            rejection_prob = float(probs[0][list(model.classes_).index(1)])
        else:
            pred_val       = float(model.predict(feature)[0])
            rejection_prob = float(1.0 / (1.0 + np.exp(-pred_val)))

        # ── Label (target is Loan_Status_Rejected: 1=Rejected, 0=Approved) ───
        if rejection_prob >= 0.5:
            prediction_label = 'Rejected ❌'
            reason = f"High rejection risk ({rejection_prob:.1%} score)"
        else:
            prediction_label = 'Approved ✅'
            reason = f"Approval confidence {(1 - rejection_prob):.1%}"

        # ── Income threshold business rule ────────────────────────────────────
        income_q = thresholds.get('income_q', 0)
        if annual_income < income_q * 0.5 and 'Approved' in prediction_label:
            prediction_label = 'Rejected ❌'
            reason = f"Annual income ₹{annual_income:,.0f} is below the minimum required threshold"

        result_text = f"{prediction_label} — {reason}"

        return render_template(
            'approval.html',
            prediction_text=result_text,
            form_data=request.form.to_dict()
        )

    except ValueError as e:
        return render_template(
            'approval.html',
            prediction_text=f"Input error: {e}",
            form_data=request.form.to_dict()
        )


if __name__ == "__main__":
    app.run(debug=True)
