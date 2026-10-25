"""Train the Random Forest and save everything the web app needs.

Usage:   python train_model.py data/cleaned_file.xlsx
         (.csv also works)
Creates: model/random_forest.joblib   trained pipeline (encoding + Random Forest)
         model/metrics.json           REAL measured metrics (test split + 5-fold CV)
         model/vehicle_counts.json    records per vehicle type
         model/analytics.json         chart data computed from your dataset
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_features import build_features, CAT_COLS, NUM_COLS, VEHICLES, CAPACITY

OUT = Path(__file__).with_name("model")
OUT.mkdir(exist_ok=True)
CROWD = ["Low", "Medium", "High", "Very High"]
RANDOM_STATE = 42


def crowd_class(values, cuts):
    return np.digitize(values, cuts)  # 0..3


def make_pipeline(regressor=None):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    rf = regressor if regressor is not None else RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("rf", rf)])


path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cleaned_file.xlsx")
df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
df.columns = [c.strip().lower() for c in df.columns]
df = df.dropna(subset=["seats_booked"]).reset_index(drop=True)  # duplicates kept: identical rows can be genuine bookings
print(f"Loaded {len(df):,} rows from {path}")

X = build_features(df)
y = df["seats_booked"].astype(float)

# Capacity per vehicle = max seats ever booked for that vehicle type in the data
veh = X["vehicle_type"]
capacity = {v: int(y[veh == v].max()) for v in sorted(veh.unique())}
cap_series = veh.map(capacity)
occ = y / cap_series  # occupancy 0..1

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
occ_te = occ.loc[y_te.index]

pipe = make_pipeline()
pipe.fit(X_tr, y_tr)

pred = pipe.predict(X_te)
mae = mean_absolute_error(y_te, pred)
rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
r2 = r2_score(y_te, pred)
# shuffled folds: the dataset is sorted by vehicle type, so plain folds would be unrepresentative
cv = cross_val_score(make_pipeline(), X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")
# R2 inside the 50-seat vehicles only (shows skill beyond "cars carry fewer people than buses")
bus_mask = (X_te["vehicle_type"].map(capacity) >= 50).values
r2_bus = r2_score(y_te[bus_mask], pred[bus_mask])
mae_bus = mean_absolute_error(y_te[bus_mask], pred[bus_mask])

# ---- Model comparison on the SAME train/test split -----------------------------
def score(name, p):
    m = (X_te["vehicle_type"].map(capacity) >= 50).values
    return {"name": name, "mae": round(float(mean_absolute_error(y_te, p)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, p))), 3),
            "r2": round(float(r2_score(y_te, p)), 4),
            "r2_bus": round(float(r2_score(y_te[m], p[m])), 4)}

veh_mean = y_tr.groupby(X_tr["vehicle_type"]).mean()
comparison = [score("Baseline (average per vehicle type)", X_te["vehicle_type"].map(veh_mean).values)]
for name, reg in [("Linear Regression", LinearRegression()),
                  ("Decision Tree", DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_STATE)),
                  ("Gradient Boosting", GradientBoostingRegressor(random_state=RANDOM_STATE))]:
    comparison.append(score(name, make_pipeline(reg).fit(X_tr, y_tr).predict(X_te)))
comparison.append(score("Random Forest (used in app)", pred))
for c in comparison:
    print(f"  {c['name']:<38} MAE={c['mae']:<6} R2={c['r2']:<7} R2(bus)={c['r2_bus']}")

# Crowd level = occupancy (seats booked / vehicle capacity): <25% Low, <50% Medium, <75% High, else Very High
cuts = np.array([0.25, 0.50, 0.75])
pred_occ = np.clip(pred / X_te["vehicle_type"].map(capacity).values, 0, 1)
true_c, pred_c = crowd_class(occ_te.values, cuts), crowd_class(pred_occ, cuts)
acc = float((true_c == pred_c).mean())
cm = confusion_matrix(true_c, pred_c, labels=[0, 1, 2, 3]).tolist()

# Feature importance, grouped back to the original columns
names = pipe.named_steps["prep"].get_feature_names_out()
imp = {}
for n, v in zip(names, pipe.named_steps["rf"].feature_importances_):
    base = n.split("__", 1)[1]
    key = next((c for c in CAT_COLS if base.startswith(c + "_")), base)
    imp[key] = imp.get(key, 0) + float(v)
imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

metrics = {
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset_file": path.name, "rows": int(len(df)),
    "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "model": "RandomForestRegressor (n_estimators=200, min_samples_leaf=2)",
    "target": "seats_booked",
    "mae": round(float(mae), 3), "rmse": round(rmse, 3), "r2": round(float(r2), 4),
    "r2_bus_only": round(float(r2_bus), 4), "mae_bus_only": round(float(mae_bus), 3),
    "cv_r2_mean": round(float(cv.mean()), 4), "cv_r2_std": round(float(cv.std()), 4),
    "crowd_accuracy": round(acc, 4), "crowd_labels": CROWD,
    "crowd_cutoffs": [25, 50, 75],
    "capacity": capacity,
    "comparison": comparison,
    "confusion_matrix": cm,
    "feature_importance": {k: round(v, 4) for k, v in imp.items()},
}
joblib.dump(pipe, OUT / "random_forest.joblib", compress=3)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ---- Fare table: median ticket_price per route + vehicle (from your data) ------
fare_df = df.assign(source=X["source"], destination=X["destination"], vehicle_type=veh)
fares = fare_df.groupby(["source", "destination", "vehicle_type"])["ticket_price"].median()
fare_json = {f"{a}|{b}|{c}": float(v) for (a, b, c), v in fares.items()}
(OUT / "fares.json").write_text(json.dumps(fare_json))

# ---- Vehicle records table ----------------------------------------------------
vc = df["vehicle_type"].astype(str).str.strip().value_counts()
rows = [{"vehicle": v, "records": int(vc.get(v, 0))} for v in VEHICLES]
rows += [{"vehicle": v, "records": int(n)} for v, n in vc.items() if v not in VEHICLES]  # any extra names
total = int(sum(r["records"] for r in rows))
for r in rows:
    r["percent"] = round(100 * r["records"] / total, 1) if total else 0
(OUT / "vehicle_counts.json").write_text(json.dumps({"rows": rows, "total": total}, indent=2))

# ---- Analytics charts (real values from the dataset) ----------------------------
ab = {"Durg": "Dg", "Bhilai": "Bh", "Raipur": "Rp", "Bilaspur": "Bl", "Rajnandgaon": "Rj"}
d = df.copy()
d["_h"] = X["hour"]
d["_wk"] = X["is_weekend"]
d["_hol"] = X["holiday"]
d["_occ"] = occ.values * 100
route = d.groupby(["source", "destination"])["_occ"].mean().sort_values(ascending=False).head(6)
hour = d.groupby("_h")["_occ"].mean()
weather = d.groupby(d["weather"].astype(str).str.strip())["_occ"].mean()
daytype = pd.Series({
    "Weekday": d.loc[(d._wk == 0) & (d._hol == 0), "_occ"].mean(),
    "Weekend": d.loc[d._wk == 1, "_occ"].mean(),
    "Holiday": d.loc[d._hol == 1, "_occ"].mean(),
}).dropna()


def hour_label(h):
    return f"{(h % 12) or 12}{'AM' if h < 12 else 'PM'}"


analytics = {
    "route": {"labels": [f"{ab.get(s, s)}→{ab.get(t, t)}" for s, t in route.index],
              "values": [round(v, 1) for v in route.values]},
    "hour": {"labels": [hour_label(int(h)) for h in hour.index], "values": [round(v, 1) for v in hour.values]},
    "weather": {"labels": list(weather.index), "values": [round(v, 1) for v in weather.values]},
    "daytype": {"labels": list(daytype.index), "values": [round(v, 1) for v in daytype.values]},
    "rows": int(len(df)),
}
(OUT / "analytics.json").write_text(json.dumps(analytics, indent=2, ensure_ascii=False))

print(f"Bus-type only: R2={r2_bus:.4f} MAE={mae_bus:.3f}")
print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}  CV R2={cv.mean():.4f}+/-{cv.std():.4f}  Crowd accuracy={acc:.2%}")
print("Vehicle records:", {r['vehicle']: r['records'] for r in rows}, "Total:", total)
print("Capacity:", capacity)
print("Saved to", OUT)
