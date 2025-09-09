# Tram Auxiliary-Energy Savings (Fixed vs Sensor)

Simulate tram trips, compute auxiliary energy for **fixed** vs **sensor-based** controls (lighting, ventilation, HVAC), and
train a small ML model to **predict % savings** from trip conditions.
## Report
**Project report (PDF):** [tram-energy-report.pdf](docs/tram-energy-report.pdf)
## Repo layout
- `generate_dataset.py` — makes `synthetic_trips.csv` with per-trip energy + `% savings`
- `train_model.py` — trains a Random Forest → saves to `models/savings_model.joblib`
- `predict_cli.py` — ask for season/hour/temp/etc → prints **ML** and **formula** % savings
- `requirements.txt` — Python packages

## Quickstart

> Python 3.10+ recommended (works with 3.11/3.12/3.13).

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python generate_dataset.py     # creates synthetic_trips.csv
python train_model.py          # trains and saves models/savings_model.joblib
python predict_cli.py          # interactive predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python generate_dataset.py
python train_model.py
python predict_cli.py
=== Tram Aux-Energy Savings Estimator ===
Season (winter/spring/summer/autumn): winter
Hour [0-23]: 7
Trip duration (minutes): 30
Outside temperature (°C): 2
Passenger count [0-200]: 140

Percent savings (Sensor vs Fixed):
  ML model:        12.4 %
  Formula (true):  12.1 %
