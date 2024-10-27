import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model and training data
model = joblib.load('rfmodel.pkl')
data = pd.read_csv('Training.csv')

# Extract the symptom columns
symptoms_columns = data.drop(columns=['prognosis']).columns

# Load the label encoder if you used it to encode the target variable
label_encoder = joblib.load('label_encoder.pkl')  # Ensure this file is saved if you used encoding

# Function to predict disease based on symptoms
def predict_disease(selected_symptoms):
    # Create a zeroed-out dataframe with columns for each symptom
    input_data = pd.DataFrame(0, index=[0], columns=symptoms_columns)

    # Set columns corresponding to the selected symptoms to 1
    for symptom in selected_symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert prediction back to original label if encoded
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected symptoms from form
    selected_symptoms = request.form.getlist('symptoms')
    
    # Predict disease based on selected symptoms
    predicted_disease = predict_disease(selected_symptoms)
    
    return render_template('result.html', disease=predicted_disease)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
