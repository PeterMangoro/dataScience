"""Leakage-safe feature engineering for the California Housing model.

Copied verbatim from assignment1/gdp_california_housing.ipynb so the export
script and the serving app compute identical features. Train-only constants
(caps, geo center) are passed in rather than hardcoded.
"""

import numpy as np
import pandas as pd

RAW_COLS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

ENG_COLS = [
    "logMedInc",
    "AveRooms_capped",
    "AveOccup_capped",
    "RoomsPerOccup",
    "logMedInc_x_RoomsPerOccup",
    "LatLon",
    "GeoDist",
]

# 15 columns: 8 raw (AveRooms/AveOccup kept uncapped) + 7 engineered.
FEATURE_ORDER = RAW_COLS + ENG_COLS


def add_engineered_features(
    X: pd.DataFrame, *, rooms_cap: float, occup_cap: float, lat0: float, lon0: float
) -> pd.DataFrame:
    X = X.copy()
    eps = 1e-6

    X["logMedInc"] = np.log1p(X["MedInc"])

    X["AveRooms_capped"] = np.minimum(X["AveRooms"], rooms_cap)
    X["AveOccup_capped"] = np.minimum(X["AveOccup"], occup_cap)

    X["RoomsPerOccup"] = X["AveRooms_capped"] / (X["AveOccup_capped"] + eps)

    X["logMedInc_x_RoomsPerOccup"] = X["logMedInc"] * X["RoomsPerOccup"]

    X["LatLon"] = X["Latitude"] * X["Longitude"]
    X["GeoDist"] = (X["Latitude"] - lat0) ** 2 + (X["Longitude"] - lon0) ** 2

    return X[FEATURE_ORDER]
