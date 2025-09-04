"""
Synthetic tram auxiliary-energy dataset generator
- For each trip, we simulate conditions (season, daylight, duration, outside temp, passengers)
- Then we compute energy for two scenarios: Fixed system vs Sensor-based system
- Outputs a CSV you can use for EDA and ML.

You can tweak the numbers in the CONFIG section to explore different assumptions.
"""

import numpy as np #for random numbers and math
import pandas as pd #for data tables and saving csv

# -------------------------
# CONFIG (tweak these later)
# -------------------------
SEASONS = ["winter", "spring", "summer", "autumn"] #menu of possible seasons
SEASON_WEIGHTS = [0.28, 0.22, 0.28, 0.22]   # probability of drawing each season

CAPACITY = 200 #tram capacity (can be changed)
COMFORT_TEMP = 21.0  # °C (target cabin temperature)

# Daylight windows by season (24h clock, hour inclusive of start, exclusive of end)
# Used only to decide if it's daylight (1) or dark (0); we don't store the hour in the dataset.
"""This dictionary encodes “roughly how long the days are” in each UK season.
It helps the simulator decide whether a trip at a random hour happens in daylight (so interior lighting energy
is lower) or in darkness (so interior lighting energy is higher)."""
DAYLIGHT_WINDOWS = {
    "winter":  (8, 16),  # ~8:00 to ~16:00
    "spring":  (6, 19),
    "summer":  (5, 21),
    "autumn":  (7, 18),
}

# Lighting (interior) coefficients (kW equivalents)
""""FIXED SYSTEM"""
L_BASE_NIGHT = 6.0     # fixed brightness at night
L_BASE_DAY   = 3.5     # fixed brightness in daytime
""""sensor based system"""
L_MIN        = 1.2     # safety minimum only in sensor mode
ALPHA_OCC    = 0.02    # energy needed with each passanger enetering the tram (so if the tram is at min when the passanger enters- each human will add 20 watt of more lighting to stay in the tram safely fro reading or moving around)
#Occupancy add on Beta_Occ
BETA_DAY     = 0.8     # dim the lights if its day time since the sun helps and already illuminates

# Ventilation (fans) coefficients (kW)
""""FIXED SYSTEM"""
VENT_FIXED    = 3.0    # fixed minimum ventilation power
""""sensor based system"""
VENT_MIN_LOW  = 1.2    # lower minimum when empty (sensor mode)
#occupancy add on Gamma_Occ
GAMMA_OCC     = 0.015  # add this much per person 

# HVAC (heating/cooling) coefficients (kW per °C and baseline kW) del T is how far we are from comfy temperature ("21")
#for every 1 degree away from 21 degree add 0.13 (HVAC_A_FIXED)
HVAC_A_FIXED  = 0.13   # kW per degree of ΔT (fixed)
HVAC_B_FIXED  = 0.7    # baseline kW (fixed, always on fans etc)
HVAC_A_SENS   = 0.11   # kW per degree of ΔT (sensor mode, slightly lower)
#Per 1 °C away from 21 °C, add 0.11 kW (a bit more efficient than 0.13)
HVAC_B_SENS   = 0.4    # lower baseline when empty (sensor)
HVAC_ETA_OCC  = 0.018  # adding kW per passenger in sensor mode (occupant load)
EXTREME_LOW   = 3.0    # °C threshold for very cold
EXTREME_HIGH  = 28.0   # °C threshold for very hot
#we assume that sensor based (1.05) it smarter and get lower penalty than fixed system (1.10)
EXTREME_MULT_FIXED = 1.10  # extra effort on extreme days (fixed system)
EXTREME_MULT_SENS  = 1.05  # smaller bump for sensor system


# Duration (minutes) distribution
DURATION_MEAN = 35
DURATION_SD   = 12
DURATION_MIN  = 10
DURATION_MAX  = 60

# Outside temperature by season (°C), simple normal approximations
TEMP_BY_SEASON = {
    "winter": (5, 4), #(mean, standard deviation)
    "spring": (12, 5),
    "summer": (24, 5),
    "autumn": (12, 5),
}

# Random seed for reproducibility
RNG = np.random.default_rng(7)

# -------------------------
# Helper functions
# -------------------------
def sample_season():
    return RNG.choice(SEASONS, p=np.array(SEASON_WEIGHTS) / np.sum(SEASON_WEIGHTS))
#picks a season at random using season weights as probavilites
def sample_hour():
    # 0..23 inclusive of 0, exclusive of 24
    return int(RNG.integers(0, 24))
#picks an hour

def is_daylight(season, hour):
    start, end = DAYLIGHT_WINDOWS[season]
    return 1 if (hour >= start and hour < end) else 0
#after season is picked checks the daylight window and sees is hour is in the window and then flags light or dark

def sample_duration_min():
    return int(np.clip(RNG.normal(DURATION_MEAN, DURATION_SD), DURATION_MIN, DURATION_MAX))

def sample_outside_temp(season):
    mu, sd = TEMP_BY_SEASON[season]
    return float(RNG.normal(mu, sd))
#draws outside temperature based on season
def sample_passengers(hour):
    """Simple, realistic-ish passenger model using hour for peaks.
       Morning/evening peaks heavier; nights lighter."""
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        # Peaks
        return int(np.clip(RNG.normal(140, 40), 0, CAPACITY))
    elif 10 <= hour <= 15:
        # Midday moderate
        return int(np.clip(RNG.normal(90, 35), 0, CAPACITY))
    else:
        # Night/late
        return int(np.clip(RNG.normal(30, 20), 0, CAPACITY))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# -------------------------
# Energy models (kWh)
# -------------------------
def lighting_kwh_fixed(daylight, dur_hr):
    kW = L_BASE_DAY if daylight == 1 else L_BASE_NIGHT
    return kW * dur_hr
#for fixed systems only fixed low(3.5) and high or night (6.0)* ..

def lighting_kwh_sensor(passengers, daylight, dur_hr):
    kW = L_MIN + ALPHA_OCC * passengers - BETA_DAY * daylight
    kW = max(L_MIN, kW)  # never below safety minimum
    return kW * dur_hr
#for sensor based it starts at bare min then adds tiny bit for each passanger, subtracts a chunk in daylight but neevr goes below the safe min

def vent_kwh_fixed(dur_hr):
    return VENT_FIXED * dur_hr
#in fixed systme fansa re always at const level

def vent_kwh_sensor(passengers, dur_hr):
    kW = VENT_MIN_LOW + GAMMA_OCC * passengers
    kW = max(VENT_MIN_LOW, kW)  # keep minimum fresh air
    return kW * dur_hr
#keeps a bare min then adds with each person

def hvac_kwh_fixed(outside_temp_c, dur_hr):
    dT = abs(COMFORT_TEMP - outside_temp_c)
    kW = HVAC_A_FIXED * dT + HVAC_B_FIXED
    if outside_temp_c < EXTREME_LOW or outside_temp_c > EXTREME_HIGH:
        kW *= EXTREME_MULT_FIXED
    return kW * dur_hr
#depeneds on how far the outside temp is from comfy (21 degree)Bigger gap (ΔT) → more work. On very cold/hot days, add a small extra penalty.

def hvac_kwh_sensor(passengers, outside_temp_c, dur_hr):
    dT = abs(COMFORT_TEMP - outside_temp_c)
    kW = HVAC_A_SENS * dT + HVAC_B_SENS + HVAC_ETA_OCC * passengers
    if outside_temp_c < EXTREME_LOW or outside_temp_c > EXTREME_HIGH:
        kW *= EXTREME_MULT_SENS
    kW = max(0.0, kW)
    return kW * dur_hr
#bit more efficent then fixed as works based on occupancy and the penalty is lower than fixed
#THE sensor based HVAC works  based on occupancy and outside temp so its smarter
# -------------------------
# Main generator
# -------------------------
"""n fake tram trips are made and for each
a random season, time, daylight,outside temp and passangers are picked
eenrgy used by fixed and sensor based systems is calculated
works out % savings"""
def generate_dataset(N=5000): 
    rows = []
    for i in range(1, N + 1):
        trip_id = f"T{i:05d}" #5 digit ID
        season = sample_season()
        hour = sample_hour()                 # used internally for daylight & passengers
        daylight = is_daylight(season, hour) # 1=light, 0=dark
        duration_min = sample_duration_min()
        dur_hr = duration_min / 60.0
        outside_temp_c = sample_outside_temp(season)
        passengers = sample_passengers(hour)
        occupancy_pct = 100.0 * passengers / CAPACITY

        # Fixed scenario
        light_f = lighting_kwh_fixed(daylight, dur_hr)
        vent_f  = vent_kwh_fixed(dur_hr)
        hvac_f  = hvac_kwh_fixed(outside_temp_c, dur_hr)
        total_f = light_f + vent_f + hvac_f

        # Sensor scenario
        light_s = lighting_kwh_sensor(passengers, daylight, dur_hr)
        vent_s  = vent_kwh_sensor(passengers, dur_hr)
        hvac_s  = hvac_kwh_sensor(passengers, outside_temp_c, dur_hr)
        total_s = light_s + vent_s + hvac_s

        # Savings (%)
        if total_f > 1e-9: #using a super small number but cant use 0
            #total_s is total auxillary energy for fixed system on a specific trip
            #total_f is total auxillary eenrgy for sesnore based system for a specific trip
            pct_savings = 100.0 * (1.0 - (total_s / total_f))
        else:
            pct_savings = np.nan #not a valid number
        
        """"build a dictornary t0
          store inputs/conditions (season, daylight, duration, temperature, passengers, occupancy),
          the energy breakdowns (lighting/vent/HVAC) for fixed and sensor,
          totals for each system,
          and the % savings."""

        rows.append({
            "trip_id": trip_id,
            "season": season,
            "daylight": daylight,                 # 1=daylight, 0=dark
            "duration_min": duration_min,
            "outside_temp_c": round(outside_temp_c, 2),
            "capacity": CAPACITY,
            "passenger_count": passengers,
            "occupancy_pct": round(occupancy_pct, 1),

            "lighting_kWh_fixed": round(light_f, 3),
            "vent_kWh_fixed": round(vent_f, 3),
            "hvac_kWh_fixed": round(hvac_f, 3),
            "total_aux_kWh_fixed": round(total_f, 3),

            "lighting_kWh_sensor": round(light_s, 3),
            "vent_kWh_sensor": round(vent_s, 3),
            "hvac_kWh_sensor": round(hvac_s, 3),
            "total_aux_kWh_sensor": round(total_s, 3),

            "pct_savings": round(pct_savings, 2)
        })

    df = pd.DataFrame(rows) #rows=list of trrip dictionaries, pd.dataFrame converts that lsit into tables
    return df

if __name__ == "__main__":
    df = generate_dataset(N=5000)
    outpath = "synthetic_trips.csv" #filename
    df.to_csv(outpath, index=False)
    print(f"Saved {outpath} with {len(df)} rows")
    # Quick sanity checks
    print(df.head(5)) #only show first five rows
    print("\nAverage totals (kWh):") #look at columns of fixed and sensors then take mean of all trips
    print(df[["total_aux_kWh_fixed", "total_aux_kWh_sensor"]].mean().round(3))
    print("\nAverage % savings overall:", round(df["pct_savings"].mean(), 2), "%")
    print("\nAverage % savings by season:")
    print(df.groupby("season")["pct_savings"].mean().round(2))
