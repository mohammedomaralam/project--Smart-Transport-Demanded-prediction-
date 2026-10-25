"""Shared feature engineering. Used by BOTH train_model.py and app.py so that
training and prediction always see exactly the same features."""
import pandas as pd

CITIES = ["Durg", "Bhilai", "Raipur", "Bilaspur", "Rajnandgaon"]
WEATHER = ["Sunny", "Cloudy", "Rainy"]

# Route distances in km
ROUTES = {
    ("Durg", "Raipur"): 42, ("Raipur", "Durg"): 42,
    ("Durg", "Bilaspur"): 145, ("Bilaspur", "Durg"): 145,
    ("Durg", "Rajnandgaon"): 70, ("Rajnandgaon", "Durg"): 70,
    ("Raipur", "Bilaspur"): 125, ("Bilaspur", "Raipur"): 125,
    ("Raipur", "Rajnandgaon"): 85, ("Rajnandgaon", "Raipur"): 85,
    ("Bilaspur", "Rajnandgaon"): 190, ("Rajnandgaon", "Bilaspur"): 190,
    # Bhilai distances are approximate road distances - edit if your project uses other values
    ("Durg", "Bhilai"): 12, ("Bhilai", "Durg"): 12,
    ("Bhilai", "Raipur"): 30, ("Raipur", "Bhilai"): 30,
    ("Bhilai", "Rajnandgaon"): 60, ("Rajnandgaon", "Bhilai"): 60,
    ("Bhilai", "Bilaspur"): 120, ("Bilaspur", "Bhilai"): 120,
}

# Vehicle -> fare rate per km (display order = order shown in the records table)
VEHICLES = {
    "Mini Bus": 4.0, "Bus": 5.0, "AC Bus": 6.0, "Sleeper": 7.0,
    "Car": 12.0, "AC Car": 15.0, "Scooter": 3.0, "Bike": 2.0,
}

# Seat capacity per vehicle (max seats_booked seen in the dataset; overwritten by training output)
CAPACITY = {"Mini Bus": 50, "Bus": 50, "AC Bus": 50, "Sleeper": 50,
            "Car": 4, "AC Car": 4, "Scooter": 2, "Bike": 2}

CAT_COLS = ["source", "destination", "vehicle_type", "weather"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend", "holiday", "rush_hour", "distance"]
FEATURES = CAT_COLS + NUM_COLS


def _flag(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "1.0", "yes", "y", "true", "holiday"]).astype(int)


def parse_dates(series):
    """Accepts dd-mm-yyyy (dataset) and yyyy-mm-dd (web form) in the same column."""
    s = series.astype(str).str.strip().str.slice(0, 10)
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    out = pd.to_datetime(s.where(iso), format="%Y-%m-%d", errors="coerce")
    out = out.fillna(pd.to_datetime(s.where(~iso), format="%d-%m-%Y", errors="coerce"))
    if out.isna().any():
        raise ValueError("Unrecognised travel_date format (use dd-mm-yyyy or yyyy-mm-dd)")
    return out


def build_features(df):
    """df needs: travel_date, travel_time, source, destination, vehicle_type, weather, holiday"""
    out = pd.DataFrame(index=df.index)
    date = parse_dates(df["travel_date"])
    out["source"] = df["source"].astype(str).str.strip()
    out["destination"] = df["destination"].astype(str).str.strip()
    out["vehicle_type"] = df["vehicle_type"].astype(str).str.strip()
    out["weather"] = df["weather"].astype(str).str.strip()
    out["hour"] = df["travel_time"].astype(str).str.extract(r"(\d{1,2}):")[0].astype(int)
    out["day_of_week"] = date.dt.weekday
    out["month"] = date.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["holiday"] = _flag(df["holiday"])
    # Rush hour: 7-10 AM and 5-8 PM
    out["rush_hour"] = (out["hour"].between(7, 10) | out["hour"].between(17, 20)).astype(int)
    out["distance"] = [ROUTES.get((s, d), 100) for s, d in zip(out["source"], out["destination"])]
    return out[FEATURES]
