from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel


# *args and **kwargs-dicotionary type 


app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "Loan Prediction API is running!"}

# model = joblib.load("../Models/loan_prediction_model.pkl")

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE_PATH = BASE_DIR / "Models" / "loan_prediction_pipeline.pkl"

pipeline = joblib.load(PIPELINE_PATH)


# CHATGPT DONT KNOW LOGIC ITNA WELL BELOW THIS
class LoanInput(BaseModel):

    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str

    ApplicantIncome: float
    CoapplicantIncome: float

    LoanAmount: float
    Loan_Amount_Term: float

    Credit_History: float

    Property_Area: str


@app.get("/")
def home():
    return {"message": "Loan Prediction API is running!"}


@app.post("/predict")
def predict(data: LoanInput):

    input_data = pd.DataFrame([data.model_dump()])

    # Feature Engineering (same as training notebook)
    input_data["Total_Income"] = (
        input_data["ApplicantIncome"] +
        input_data["CoapplicantIncome"]
    )

    input_data["Family_Size"] = np.where(
        input_data["Married"] == "Yes",
        input_data["Dependents"] + 2,
        input_data["Dependents"] + 1
    )

    input_data["Income_per_member"] = (
        input_data["Total_Income"] /
        input_data["Family_Size"]
    )

    prediction = pipeline.predict(input_data)

    return {
        "Prediction": "Approved" if prediction[0] == "Y" else "Rejected"
    }