from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd


# Load the model, columns and scaler
model = joblib.load('models/predictive_maintenance_model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

# API definition
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = request.form.to_dict()

    # Convert form data to dataframe
    input_data = pd.DataFrame([form_data])

    # Ensure correct data types
    input_data = input_data.astype({
        'Type': 'object',
        'Air_temperature': 'float64',
        'Process_temperature': 'float64',
        'Rotational_speed': 'int64',
        'Torque': 'float64',
        'Tool_wear': 'int64',
        'TWF': 'int64',
        'HDF': 'int64',
        'PWF': 'int64',
        'OSF': 'int64',
        'RNF': 'int64'
    })

    # One-hot encode the 'Type' feature
    input_data = pd.get_dummies(input_data, columns=['Type'])

    # Align the input data columns with the training data columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Map the prediction to a readable format
    prediction_label = 'Machine will fail' if prediction[0] == 1 else 'Machine will not fail'

    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
