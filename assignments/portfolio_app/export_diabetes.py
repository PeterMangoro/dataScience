"""Train and export the Pima Diabetes classifier for serving.

Reproduces the winning pipeline from
assignment2/pima_diabetes_assignment2.ipynb: clinical zeros recoded to NaN
(outside the pipeline), then SimpleImputer(median) -> StandardScaler ->
balanced LogisticRegression.

Run:
    python assignments/portfolio_app/export_diabetes.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR.parent / "assignment2" / "diabetes.csv"
OUT = APP_DIR / "models" / "diabetes_pipeline.joblib"

RANDOM_STATE = 9
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
CLINICAL_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)
    X = df[FEATURES].copy()
    y = df["Outcome"].copy()

    # Clinical zeros are implausible measurements -> treat as missing.
    X[CLINICAL_ZERO_AS_MISSING] = X[CLINICAL_ZERO_AS_MISSING].astype("float64")
    X[CLINICAL_ZERO_AS_MISSING] = X[CLINICAL_ZERO_AS_MISSING].mask(
        X[CLINICAL_ZERO_AS_MISSING] == 0, np.nan
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"Test ROC-AUC: {test_auc:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features": FEATURES,
            "clinical_zero_as_missing": CLINICAL_ZERO_AS_MISSING,
        },
        OUT,
    )
    size_kb = OUT.stat().st_size / 1024
    print(f"Saved pipeline to: {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
