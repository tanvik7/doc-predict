from flask import Flask, render_template, request, send_file
import pandas as pd
import os
from ml import load_and_process_data, train_ml_model, get_best_doctors, export_to_csv

df = load_and_process_data("doc-predict/dummy_npi_data.xlsx")
model = train_ml_model(df)
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    survey_time = int(request.form["survey_time"])
    best_doctors = get_best_doctors(df, model, survey_time)

    if best_doctors.empty:
        return "No doctors available at this time."

    output_file = "best_doctors.csv"
    export_to_csv(best_doctors, output_file)
    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

