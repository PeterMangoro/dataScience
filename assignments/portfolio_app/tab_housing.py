"""California Housing regression tab.

Loads the exported Eng RidgeCV bundle and predicts median house value from the
8 raw census block-group features (engineered server-side).
"""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from housing_features import RAW_COLS, add_engineered_features

MODEL_PATH = Path(__file__).resolve().parent / "models" / "housing_model.joblib"

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_consts = _bundle["consts"]


def predict(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    row = pd.DataFrame(
        [[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]],
        columns=RAW_COLS,
    )
    X_eng = add_engineered_features(row, **_consts)
    pred = float(_model.predict(X_eng)[0])
    dollars = max(pred, 0.0) * 100_000
    return f"Predicted median home value: ${dollars:,.0f}"


def build_housing_tab() -> gr.Interface:
    inputs = [
        gr.Slider(0.5, 15.0, value=3.5, step=0.1, label="Median income (tens of thousands $)"),
        gr.Slider(1, 52, value=29, step=1, label="Median house age (years)"),
        gr.Slider(1.0, 12.0, value=5.2, step=0.1, label="Average rooms per household"),
        gr.Slider(0.5, 2.0, value=1.1, step=0.05, label="Average bedrooms per household"),
        gr.Slider(3, 12000, value=1166, step=1, label="Block-group population"),
        gr.Slider(1.0, 8.0, value=3.0, step=0.1, label="Average household occupancy"),
        gr.Slider(32.5, 42.0, value=34.0, step=0.1, label="Latitude"),
        gr.Slider(-124.3, -114.3, value=-118.5, step=0.1, label="Longitude"),
    ]
    return gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction"),
        title="California Housing - Median Value Regression",
        description=(
            "Engineered linear model (RidgeCV) on 1990 California census block groups. "
            "Adjust the predictors to estimate median home value. Validation R2 ~0.66."
        ),
        flagging_mode="never",
    )
