from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load model and scalers (correct paths for your folder)
model = load_model('models/best_model.keras')
scaler_data = joblib.load('models/scaler_data.sav')
scaler_target = joblib.load('models/scaler_target.sav')

@app.route('/')
def home():
    return render_template('patient_details.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    cholesterol = int(request.form['cholesterol'])
    bp = int(request.form['bp'])
    smoking = int(request.form['smoking'])
    diabetes = int(request.form['diabetes'])
    family_history = int(request.form['family_history'])

    # Create feature array
    features = np.array([[gender, age, cholesterol, bp, smoking, diabetes, family_history]])

    # Scale → predict → inverse scale
    features_scaled = scaler_data.transform(features)
    prediction_scaled = model.predict(features_scaled)
    risk_score = scaler_target.inverse_transform(prediction_scaled)[0][0]
    risk_score = round(risk_score, 2)

    # Choose color and message
    if risk_score < 20:
        color = "low"
        level = "Low Risk"
    elif risk_score < 40:
        color = "medium"
        level = "Moderate Risk"
    else:
        color = "high"
        level = "High Risk"

    return render_template('patient_results.html',
                           risk_score=risk_score,
                           color_class=color,
                           level=level)

if __name__ == '__main__':
    app.run(debug=True)