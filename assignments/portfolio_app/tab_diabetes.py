"""Pima Diabetes classification tab.

Loads the exported balanced-logistic pipeline and returns diabetes probability
from 8 clinical inputs. Replicates the 0 -> NaN recoding of clinical zeros,
which lives outside the pickled pipeline.
"""

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "diabetes_pipeline.joblib"

_bundle = joblib.load(MODEL_PATH)
_pipeline = _bundle["pipeline"]
_features = _bundle["features"]
_clinical = _bundle["clinical_zero_as_missing"]


def predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    row = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=_features,
    )
    row[_clinical] = row[_clinical].astype("float64").mask(row[_clinical] == 0, np.nan)
    proba = _pipeline.predict_proba(row)[0]
    return {"No diabetes": float(proba[0]), "Diabetes": float(proba[1])}


def build_diabetes_tab() -> gr.Interface:
    inputs = [
        gr.Slider(0, 17, value=3, step=1, label="Pregnancies"),
        gr.Slider(0, 200, value=117, step=1, label="Glucose (plasma concentration)"),
        gr.Slider(0, 130, value=72, step=1, label="Blood pressure (diastolic, mm Hg)"),
        gr.Slider(0, 99, value=23, step=1, label="Skin thickness (triceps, mm)"),
        gr.Slider(0, 850, value=125, step=1, label="Insulin (2-hour serum, mu U/ml)"),
        gr.Slider(0, 67, value=32, step=0.1, label="BMI"),
        gr.Slider(0.05, 2.5, value=0.37, step=0.01, label="Diabetes pedigree function"),
        gr.Slider(21, 90, value=29, step=1, label="Age (years)"),
    ]
    return gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="Pima Diabetes - Risk Classification",
        description=(
            "Balanced logistic regression (median impute -> scale -> logistic) on the "
            "Pima Indians Diabetes dataset. Clinical zeros are treated as missing. "
            "Test ROC-AUC ~0.85. Screening demo only - not medical advice."
        ),
        flagging_mode="never",
    )
