# train_model.py
# Trains a Random Forest Regressor to predict % savings from trip conditions.
# Saves the trained model to models/savings_model.joblib and prints simple metrics.

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_dataset import generate_dataset  # <- adjust if your file name differs

DATA_PATH = "synthetic_trips.csv"
MODEL_PATH = "models/savings_model.joblib"

def load_or_make_data(N=5000) -> pd.DataFrame:
    """Load CSV if present; otherwise generate it once."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset(N=N)
        df.to_csv(DATA_PATH, index=False)
    # drop rows where target is NaN (safety)
    return df.dropna(subset=["pct_savings"]).reset_index(drop=True)

def build_pipeline() -> Pipeline:
    # Our input columns
    categorical = ["season"]  # one-hot
    numeric = ["daylight", "duration_min", "outside_temp_c", "passenger_count", "occupancy_pct"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )

    # Random Forest with beginner-friendly, stable settings
    rf = RandomForestRegressor(
        n_estimators=300,       # ~300 trees for stability
        max_features="sqrt",    # ~sqrt(total_features) per split (diversity)
        min_samples_leaf=5,     # each leaf must average ≥5 trips (smooth)
        min_samples_split=10,   # need ≥10 rows to consider splitting
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("pre", pre), ("rf", rf)])

def main():
    df = load_or_make_data(N=5000)

    features = ["season", "daylight", "duration_min", "outside_temp_c", "passenger_count", "occupancy_pct"]
    target = "pct_savings"

    X = df[features].copy()
    y = df[target].astype(float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)

    preds = pipe.predict(Xte)
    mae  = mean_absolute_error(yte, preds)
    mse = mean_squared_error(yte, preds)  # no 'squared' kwarg
    rmse = float(np.sqrt(mse))
    r2   = r2_score(yte, preds)

    print("Holdout performance (predicting % savings):")
    print(f"  MAE  = {mae:.2f} percentage points")
    print(f"  RMSE = {rmse:.2f} percentage points")
    print(f"  R^2  = {r2:.3f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH}")

    # Show simple feature importance (from the RF inside the pipeline)
    rf = pipe.named_steps["rf"]
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["season"]))
    num_names = ["daylight", "duration_min", "outside_temp_c", "passenger_count", "occupancy_pct"]
    names = cat_names + num_names
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\nTop features (model reliance):")
    for i in order:
        print(f"  {names[i]:<25s} {importances[i]:.3f}")

if __name__ == "__main__":
    main()
