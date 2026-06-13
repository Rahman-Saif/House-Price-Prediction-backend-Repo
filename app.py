import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
model_path = 'RidgeModel.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = float(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Create a DataFrame from the input data to match model's expected input
    input_df = pd.DataFrame([[location, total_sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Format the output for house price
    output = round(prediction, 2)

    return render_template('index.html', prediction_text=f'Predicted House Price: {output} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)