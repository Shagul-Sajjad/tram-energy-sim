# predict_cli.py
# Simple interactive CLI: enter trip conditions -> get % savings.
# Prints both the ML prediction and the formula ("physics") result.

import joblib
import pandas as pd

from generate_dataset import (
    CAPACITY, is_daylight,
    lighting_kwh_fixed, vent_kwh_fixed, hvac_kwh_fixed,
    lighting_kwh_sensor, vent_kwh_sensor, hvac_kwh_sensor,
)

MODEL_PATH = "models/savings_model.joblib"

def physics_pct_savings(season: str, hour: int, duration_min: int, outside_temp_c: float, passengers: int) -> float:
    """Exact % savings from your original formulas."""
    daylight = is_daylight(season, hour)
    dur_hr = duration_min / 60.0

    # Fixed totals
    lf = lighting_kwh_fixed(daylight, dur_hr)
    vf = vent_kwh_fixed(dur_hr)
    hf = hvac_kwh_fixed(outside_temp_c, dur_hr)
    total_f = lf + vf + hf

    # Sensor totals
    ls = lighting_kwh_sensor(passengers, daylight, dur_hr)
    vs = vent_kwh_sensor(passengers, dur_hr)
    hs = hvac_kwh_sensor(passengers, outside_temp_c, dur_hr)
    total_s = ls + vs + hs

    if total_f <= 1e-9:
        return float("nan")
    return 100.0 * (1.0 - total_s / total_f)

def predict_ml(season: str, hour: int, duration_min: int, outside_temp_c: float, passengers: int) -> float:
    """Predict % savings using the trained Random Forest."""
    model = joblib.load(MODEL_PATH)
    daylight = is_daylight(season, hour)
    occ = 100.0 * passengers / CAPACITY
    row = pd.DataFrame([{
        "season": season,
        "daylight": daylight,
        "duration_min": duration_min,
        "outside_temp_c": outside_temp_c,
        "passenger_count": passengers,
        "occupancy_pct": occ,
    }])
    return float(model.predict(row)[0])

def main():
    print("=== Tram Aux-Energy Savings Estimator ===")
    season = input("Season (winter/spring/summer/autumn): ").strip().lower()
    hour = int(input("Hour [0-23]: ").strip())
    duration_min = int(input("Trip duration (minutes): ").strip())
    outside_temp_c = float(input("Outside temperature (°C): ").strip())
    passengers = int(input(f"Passenger count [0-{CAPACITY}]: ").strip())

    ml  = predict_ml(season, hour, duration_min, outside_temp_c, passengers)
    phy = physics_pct_savings(season, hour, duration_min, outside_temp_c, passengers)

    print("\nPercent savings (Sensor vs Fixed):")
    print(f"  ML model:        {ml:.2f} %")
    print(f"  Formula (true):  {phy:.2f} %")
    print("\nNote: With synthetic training data, these should be close. "
          "With real data later, ML can learn patterns beyond the simple formula.")

if __name__ == "__main__":
    main()
