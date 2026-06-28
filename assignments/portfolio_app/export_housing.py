"""Train and export the California Housing regression model for serving.

Reproduces the winning "Eng RidgeCV" pipeline from
assignment1/gdp_california_housing.ipynb: leakage-safe engineered features
(train-only caps + geo center) fed to StandardScaler -> RidgeCV.

Run:
    python assignments/portfolio_app/export_housing.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from housing_features import FEATURE_ORDER, add_engineered_features

APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "models" / "housing_model.joblib"

RANDOM_STATE = 42
ALPHA_GRID = np.logspace(-4, 2, 30)


def main() -> None:
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rooms_cap = float(np.quantile(X_train["AveRooms"], 0.99))
    occup_cap = float(np.quantile(X_train["AveOccup"], 0.99))
    lat0 = float(X_train["Latitude"].mean())
    lon0 = float(X_train["Longitude"].mean())
    consts = {"rooms_cap": rooms_cap, "occup_cap": occup_cap, "lat0": lat0, "lon0": lon0}
    print(f"Train-only constants: {consts}")

    X_eng_train = add_engineered_features(X_train, **consts)
    X_eng_valid = add_engineered_features(X_valid, **consts)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=ALPHA_GRID, cv=cv)),
        ]
    )
    model.fit(X_eng_train, y_train)

    pred = model.predict(X_eng_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    r2 = float(r2_score(y_valid, pred))
    print(f"Validation RMSE: {rmse:.4f} | R2: {r2:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "consts": consts, "feature_order": FEATURE_ORDER}, OUT
    )
    size_kb = OUT.stat().st_size / 1024
    print(f"Saved model to: {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
