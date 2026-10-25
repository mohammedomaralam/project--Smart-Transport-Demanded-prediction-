# Smart Transport Demand Prediction — Flask app + Random Forest

Cities: Durg, Bhilai, Raipur, Bilaspur, Rajnandgaon. Dataset: 12,000 records, 8 vehicle types. Target variable: `seats_booked`.

## Folder layout
```
app.py              Flask app (loads the trained model automatically)
ml_features.py      Feature engineering shared by training AND the app
train_model.py      Trains the Random Forest, saves model + real metrics
templates/ static/  Pages, CSS, JS, logo
data/               Smart_Transport_Dataset.xlsx (your dataset)
model/              Created by train_model.py
```

## Run (Windows / VS Code terminal)
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_model.py data\Smart_Transport_Dataset.xlsx
.venv\Scripts\python.exe app.py
```
Open http://127.0.0.1:5000  (the Model Status page should say "Random Forest model connected").

## What train_model.py creates in model/
- `random_forest.joblib`  – full pipeline (one-hot encoding + Random Forest)
- `metrics.json`          – MAE, RMSE, R², 5-fold CV R², crowd accuracy, confusion matrix, feature importance
- `metrics.json` also holds the model comparison (Baseline, Linear Regression, Decision Tree, Gradient Boosting, Random Forest)
- `vehicle_counts.json`   – records per vehicle type (Mini Bus, Bus, AC Bus, Sleeper, Car, AC Car, Scooter, Bike, Total)
- `analytics.json`        – chart values computed from your dataset

## How a prediction works
1. Booking form -> same features as training (hour, day_of_week, month, is_weekend, holiday, rush_hour, distance, source, destination, vehicle_type, weather).
2. The Random Forest predicts expected `seats_booked`.
3. Demand % = predicted seats / vehicle capacity (capacity = max seats seen per vehicle: 50 for Mini Bus/Bus/AC Bus/Sleeper, 4 for Car/AC Car, 2 for Scooter/Bike).
   Crowd level: Low < 25%, Medium < 50%, High < 75%, Very High above.
4. Fare = median `ticket_price` for that route + vehicle from the dataset (x passengers for buses; per trip for cars/bikes/scooters).

If `model/` is empty, the app falls back to a demo formula and says so on the booking and Model Status pages.

## Extra features
- **Quieter departure times:** after a prediction, the booking page predicts the same trip for every hour (05:00-22:00) and lists the 3 quietest times.
- **Model comparison table** on the Model Status page, all models on the same train/test split.

## Open on your phone with a QR code
Run `python app.py`: the terminal prints a QR code. Scan it with the phone camera (phone and laptop on the SAME Wi-Fi or phone hotspot). If Windows Firewall asks, click *Allow* for private networks.

Download the QR as a PNG without starting the server: `python make_qr.py` (saves to the project folder and your Downloads folder, then opens it).

## Admin page (who registered / logged in)
Log in as the admin and open **Admin** in the top menu (or http://127.0.0.1:5000/admin). The admin is the first registered account, or the e-mail set in `SMART_TRANSPORT_ADMIN_EMAIL`. Login history is recorded from the moment this version runs.

## Notes
- Bhilai road distances in `ml_features.py` are approximate - edit them if your report uses other values.
- Departures are limited to 05:00-22:59, the range present in the dataset.
- Re-run `train_model.py` any time the dataset changes; restart the app afterwards.
- Before deployment: set `SMART_TRANSPORT_SECRET_KEY`, disable debug, add CSRF protection.
