import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import openpyxl

def load_and_process_data(file_path):
    xls = pd.ExcelFile("dummy_npi_data.xlsx")
    df = pd.read_excel(xls, sheet_name="Dataset")
    df["Login Hour"] = df["Login Time"].dt.hour
    df["Logout Hour"] = df["Logout Time"].dt.hour
    df["Survey Participation Rate"] = df["Count of Survey Attempts"] / df["Usage Time (mins)"]
    df["Survey Participation Rate"].fillna(0, inplace=True)
    peak_hours = df.groupby("NPI")["Login Hour"].agg(lambda x: x.value_counts().idxmax())
    df = df.merge(peak_hours.rename("Peak Hour"), on="NPI")
    df["Survey Participation"] = (df["Count of Survey Attempts"] > 0).astype(int)
    return df

def train_ml_model(df):
    features = ["Login Hour", "Logout Hour", "Usage Time (mins)", "Peak Hour"]
    X = df[features]
    y = df["Survey Participation"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    
    ensemble_model = VotingClassifier(
        estimators=[("random_forest", rf_model), ("logistic_regression", lr_model)],
        voting='soft'
    )
    
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Model Accuracy: {accuracy:.2f}")
    return ensemble_model

def get_best_doctors(df, model, input_time):
    input_df = df.copy()
    input_df = input_df[(input_df["Login Hour"] <= input_time) & (input_df["Logout Hour"] >= input_time)]
    if input_df.empty:
        print("No doctors available at this time.")
        return pd.DataFrame()
    X_input = input_df[["Login Hour", "Logout Hour", "Usage Time (mins)", "Peak Hour"]]
    input_df["Prediction"] = model.predict(X_input)
    selected_doctors = input_df[input_df["Prediction"] == 1]
    output_columns = ["NPI", "State", "Region", "Speciality", "Survey Participation Rate"]
    return selected_doctors[output_columns]

def export_to_csv(doctors_df, output_file):
    doctors_df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")
