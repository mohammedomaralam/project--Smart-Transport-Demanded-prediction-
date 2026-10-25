from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_features import CITIES, ROUTES, VEHICLES, WEATHER, CAPACITY, build_features

app = Flask(__name__)
app.secret_key = os.environ.get("SMART_TRANSPORT_SECRET_KEY", "local-demo-key-change-before-deployment")
BASE = Path(__file__).parent
DB = BASE / "transport.db"
MODEL_DIR = BASE / "model"

# ----------------------------------------------------------------------------
# Model loading (real Random Forest if model/ files exist, otherwise demo formula)
# ----------------------------------------------------------------------------
MODEL = None
META = {}
FARES = {}


def load_json(name):
    p = MODEL_DIR / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupted file
            print(f"[warn] could not read {name}: {exc}")
    return None


def load_model():
    global MODEL, META, FARES
    FARES = load_json("fares.json") or {}
    path = MODEL_DIR / "random_forest.joblib"
    META = load_json("metrics.json") or {}
    if path.exists() and META:
        try:
            import joblib
            MODEL = joblib.load(path)
            print("[ok] Random Forest model loaded")
        except Exception as exc:
            MODEL = None
            print(f"[warn] model file found but could not be loaded: {exc}")


load_model()

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        source TEXT NOT NULL,
        destination TEXT NOT NULL,
        distance REAL NOT NULL,
        travel_date TEXT NOT NULL,
        travel_time TEXT NOT NULL,
        weather TEXT NOT NULL,
        vehicle TEXT NOT NULL,
        passengers INTEGER NOT NULL,
        fare REAL NOT NULL,
        predicted_demand REAL NOT NULL,
        crowd_level TEXT NOT NULL,
        booking_status TEXT NOT NULL DEFAULT 'Confirmed',
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS login_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        logged_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()


ADMIN_EMAIL = os.environ.get("SMART_TRANSPORT_ADMIN_EMAIL", "").strip().lower()


def is_admin_user(user):
    """Admin = the e-mail in SMART_TRANSPORT_ADMIN_EMAIL, otherwise the very first registered account."""
    if ADMIN_EMAIL:
        return user["email"] == ADMIN_EMAIL
    return user["id"] == 1


def login_required():
    return "user_id" in session

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def crowd_from_demand(demand):
    # demand = occupancy % (seats booked / vehicle capacity)
    if demand < 25:
        return "Low"
    if demand < 50:
        return "Medium"
    if demand < 75:
        return "High"
    return "Very High"


def capacity_of(vehicle):
    return (META.get("capacity") or CAPACITY).get(vehicle, CAPACITY.get(vehicle, 50))


def demo_demand(distance, travel_date, travel_time, weather, vehicle, passengers):
    """Fallback only (used when no trained model is present)."""
    hour = int(travel_time.split(":")[0])
    demand = 30.0
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        demand += 25
    elif 11 <= hour <= 15:
        demand += 5
    else:
        demand -= 2
    demand += 12 if distance <= 80 else (-3 if distance >= 160 else 0)
    demand += {"Sunny": 5, "Rainy": -4}.get(weather, 0)
    if vehicle in ("AC Bus", "AC Car", "Sleeper"):
        demand += 4
    elif vehicle in ("Mini Bus", "Scooter", "Bike"):
        demand -= 4
    demand += min(passengers * 1.5, 10)
    try:
        if datetime.strptime(travel_date, "%Y-%m-%d").weekday() >= 5:
            demand += 8
    except ValueError:
        pass
    return max(5, min(95, round(demand, 1)))


def estimate_fare(source, destination, vehicle, passengers, distance):
    """Median ticket price for this route + vehicle from the dataset (per seat for buses,
    per trip for cars/bikes). Falls back to a per-km formula if the table is missing."""
    price = FARES.get(f"{source}|{destination}|{vehicle}")
    if price is None:
        return round(distance * VEHICLES.get(vehicle, 5.0) + passengers * 10, 2)
    per_trip = vehicle in ("Car", "AC Car", "Scooter", "Bike")
    return round(price if per_trip else price * passengers, 2)


def calculate_prediction(source, destination, travel_date, travel_time, weather, vehicle, passengers, holiday=0):
    distance = ROUTES.get((source, destination), 100)
    fare = estimate_fare(source, destination, vehicle, passengers, distance)
    cap = capacity_of(vehicle)
    seats = None

    if MODEL is not None:
        row = pd.DataFrame([{
            "travel_date": travel_date, "travel_time": travel_time,
            "source": source, "destination": destination,
            "vehicle_type": vehicle, "weather": weather, "holiday": int(holiday),
        }])
        seats = float(MODEL.predict(build_features(row))[0])
        demand = round(max(1.0, min(99.0, seats / cap * 100)), 1)
        engine = "Random Forest model"
    else:
        demand = demo_demand(distance, travel_date, travel_time, weather, vehicle, passengers)
        engine = "Demo formula (model not trained)"

    return {
        "distance": distance, "demand": demand, "crowd": crowd_from_demand(demand),
        "fare": fare, "seats": None if seats is None else round(seats, 1),
        "capacity": cap, "engine": engine,
    }

def quieter_times(source, destination, travel_date, travel_time, weather, vehicle, holiday, current_demand):
    """Predict the same trip for every departure hour (05-22) and return the 3 quietest."""
    if MODEL is None:
        return []
    minute = travel_time.split(":")[1]
    hours = list(range(5, 23))
    rows = pd.DataFrame([{
        "travel_date": travel_date, "travel_time": f"{h:02d}:{minute}",
        "source": source, "destination": destination,
        "vehicle_type": vehicle, "weather": weather, "holiday": int(holiday),
    } for h in hours])
    cap = capacity_of(vehicle)
    occ = np.clip(MODEL.predict(build_features(rows)) / cap * 100, 1, 99)
    current_hour = int(travel_time.split(":")[0])
    best = sorted(zip(hours, occ), key=lambda x: x[1])
    out = []
    for h, d in best:
        if h == current_hour:
            continue
        out.append({"time": f"{h:02d}:{minute}", "demand": round(float(d), 1),
                    "crowd": crowd_from_demand(d), "saving": round(current_demand - float(d), 1)})
        if len(out) == 3:
            break
    return out


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def home():
    if login_required():
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill all fields.", "error")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")
            return redirect(url_for("register"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                (name, email, generate_password_hash(password), datetime.now().isoformat(timespec="seconds"))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            flash("An account with this email already exists.", "error")
            return redirect(url_for("register"))
        conn.close()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["is_admin"] = is_admin_user(user)
            conn = get_db()
            conn.execute("INSERT INTO login_log(user_id, logged_at) VALUES(?,?)",
                         (user["id"], datetime.now().isoformat(timespec="seconds")))
            conn.commit()
            conn.close()
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()
    bookings = conn.execute(
        "SELECT * FROM bookings WHERE user_id=? ORDER BY id DESC LIMIT 5",
        (session["user_id"],)
    ).fetchall()
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM bookings WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()["c"]
    conn.close()

    return render_template("dashboard.html", bookings=bookings, booking_count=count, model_connected=MODEL is not None)


@app.route("/book", methods=["GET", "POST"])
def book():
    if not login_required():
        return redirect(url_for("login"))

    result = None
    if request.method == "POST":
        source = request.form.get("source")
        destination = request.form.get("destination")
        travel_date = request.form.get("travel_date")
        travel_time = request.form.get("travel_time")
        weather = request.form.get("weather")
        vehicle = request.form.get("vehicle")
        holiday = 1 if request.form.get("holiday") in ("1", "yes", "Yes") else 0
        try:
            passengers = int(request.form.get("passengers", 1))
        except (TypeError, ValueError):
            flash("Please choose a valid passenger count.", "error")
            return redirect(url_for("book"))

        if source not in CITIES or destination not in CITIES:
            flash("Please choose a valid route.", "error")
            return redirect(url_for("book"))
        if not travel_date or not travel_time:
            flash("Please select a travel date and time.", "error")
            return redirect(url_for("book"))
        try:
            datetime.strptime(travel_date, "%Y-%m-%d")
            datetime.strptime(travel_time, "%H:%M")
        except ValueError:
            flash("Please enter a valid travel date and time.", "error")
            return redirect(url_for("book"))
        if not 5 <= int(travel_time.split(":")[0]) <= 22:
            flash("Predictions are available for departures between 05:00 and 22:59 (the range in the dataset).", "error")
            return redirect(url_for("book"))
        if weather not in WEATHER:
            flash("Please choose a valid weather option.", "error")
            return redirect(url_for("book"))
        if vehicle not in VEHICLES:
            flash("Please choose a valid vehicle.", "error")
            return redirect(url_for("book"))
        if source == destination:
            flash("Source and destination must be different.", "error")
            return redirect(url_for("book"))
        if passengers < 1 or passengers > 10:
            flash("Passengers must be between 1 and 10.", "error")
            return redirect(url_for("book"))
        if passengers > capacity_of(vehicle):
            flash(f"A {vehicle} seats at most {capacity_of(vehicle)} passengers.", "error")
            return redirect(url_for("book"))

        p = calculate_prediction(source, destination, travel_date, travel_time,
                                 weather, vehicle, passengers, holiday)

        result = {
            "source": source, "destination": destination,
            "date": travel_date, "time": travel_time,
            "weather": weather, "vehicle": vehicle,
            "passengers": passengers, "holiday": holiday,
            "distance": p["distance"], "demand": p["demand"],
            "crowd": p["crowd"], "fare": p["fare"],
        }

        if request.form.get("confirm_booking") == "yes":
            conn = get_db()
            cur = conn.execute("""INSERT INTO bookings
                (user_id,source,destination,distance,travel_date,travel_time,weather,
                 vehicle,passengers,fare,predicted_demand,crowd_level,booking_status,created_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session["user_id"], source, destination, p["distance"], travel_date,
                 travel_time, weather, vehicle, passengers, p["fare"], p["demand"], p["crowd"],
                 "Confirmed", datetime.now().isoformat(timespec="seconds")))
            booking_id = cur.lastrowid
            conn.commit()
            conn.close()
            return redirect(url_for("booking_success", booking_id=booking_id))

        result["seats"] = p["seats"]
        result["engine"] = p["engine"]
        result["alternatives"] = quieter_times(source, destination, travel_date, travel_time,
                                               weather, vehicle, holiday, p["demand"])

    return render_template("book.html", cities=CITIES, vehicles=list(VEHICLES), weather_options=WEATHER, result=result)


@app.route("/booking/<int:booking_id>")
def booking_success(booking_id):
    if not login_required():
        return redirect(url_for("login"))
    conn = get_db()
    booking = conn.execute(
        "SELECT * FROM bookings WHERE id=? AND user_id=?",
        (booking_id, session["user_id"])
    ).fetchone()
    conn.close()
    if not booking:
        return "Booking not found", 404
    return render_template("success.html", booking=booking)


@app.route("/bookings")
def bookings():
    if not login_required():
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM bookings WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("bookings.html", bookings=rows)


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", analytics=load_json("analytics.json"))


@app.route("/models")
def models():
    vehicle_data = load_json("vehicle_counts.json")
    metrics = {k: v for k, v in META.items() if k != "quantiles"} if MODEL is not None else None
    return render_template("models.html", model_connected=MODEL is not None,
                           metrics=metrics, vehicle_data=vehicle_data)


@app.route("/admin")
def admin():
    if not login_required():
        return redirect(url_for("login"))
    if not session.get("is_admin"):
        flash("Only the admin can open this page.", "error")
        return redirect(url_for("dashboard"))
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    one = lambda q, a=(): conn.execute(q, a).fetchone()[0]
    stats = {
        "users": one("SELECT COUNT(*) FROM users"),
        "logins": one("SELECT COUNT(*) FROM login_log"),
        "logins_today": one("SELECT COUNT(*) FROM login_log WHERE logged_at LIKE ?", (today + "%",)),
        "users_today": one("SELECT COUNT(DISTINCT user_id) FROM login_log WHERE logged_at LIKE ?", (today + "%",)),
        "users_ever_logged_in": one("SELECT COUNT(DISTINCT user_id) FROM login_log"),
        "bookings": one("SELECT COUNT(*) FROM bookings"),
    }
    users = conn.execute("""
        SELECT u.id, u.name, u.email, u.created_at,
               COUNT(l.id) AS logins, MAX(l.logged_at) AS last_login,
               (SELECT COUNT(*) FROM bookings b WHERE b.user_id = u.id) AS bookings
        FROM users u LEFT JOIN login_log l ON l.user_id = u.id
        GROUP BY u.id ORDER BY u.id DESC""").fetchall()
    recent = conn.execute("""
        SELECT u.name, u.email, l.logged_at FROM login_log l
        JOIN users u ON u.id = l.user_id ORDER BY l.id DESC LIMIT 10""").fetchall()
    conn.close()
    return render_template("admin.html", stats=stats, users=users, recent=recent)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not login_required():
        return jsonify({"error": "Login required"}), 401
    data = request.get_json(force=True)
    try:
        p = calculate_prediction(
            data["source"], data["destination"], data["travel_date"],
            data["travel_time"], data.get("weather", "Sunny"),
            data.get("vehicle", "Bus"), int(data.get("passengers", 1)),
            int(data.get("holiday", 0)),
        )
        return jsonify(p)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def lan_ip():
    """IP address of this computer on the Wi-Fi / hotspot (what a phone must open)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no data is sent; just picks the active network card
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def show_qr(url):
    """Print a scannable QR code in the terminal and save it as qr_code.png."""
    try:
        import sys
        import qrcode
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        print("\n  Scan this QR code with your phone camera (phone must be on the SAME Wi-Fi):\n")
        qr.print_ascii(invert=True)
        qr.make_image().save(BASE / "qr_code.png")
        print(f"\n  Or open on phone: {url}")
        print("  (QR image also saved as qr_code.png)\n")
    except Exception as exc:
        print(f"[warn] QR code could not be shown ({exc}). Open this on your phone instead: {url}")


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # host 0.0.0.0 lets other devices on the same network (your phone) reach the app
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        show_qr(f"http://{lan_ip()}:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
    