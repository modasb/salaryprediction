import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Create flask app
app = Flask(__name__)

# Load the trained model and the preprocessing pipeline
model = joblib.load("modele_regression.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/index")
def Home():
    return render_template("index.html")


# Preprocessing function
def preprocess_input(data):
    # Convert form data to DataFrame
    df = pd.DataFrame(data, index=[0])

    # Apply the same preprocessing steps as during training
    df_pipelined = pipeline.transform(df)

    return df_pipelined


@app.route("/predict", methods=["POST"])
def predict():
    # Get the form data from the request
    form_data = request.form.to_dict()

    # Convert numeric fields to float if needed
    for key in form_data.keys():
        try:
            form_data[key] = float(form_data[key])
        except ValueError:
            pass

    # Preprocess the form data
    preprocessed_data = preprocess_input(form_data)

    # Make predictions
    prediction = model.predict(preprocessed_data)

    return render_template("index.html", prediction_text="The salary is ${:,.2f}".format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)
