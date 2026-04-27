from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import json
from datetime import datetime
import requests
import random

app = Flask(__name__)

# ─── Model & Data Loading ────────────────────────────────────────────
MODEL_PATH = "models/"
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))
explainer = joblib.load(os.path.join(MODEL_PATH, "shap_explainer.pkl"))

# Load as XGBClassifier (matches how train_and_save.py saved it)
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.load_model(os.path.join(MODEL_PATH, "xgb_model.json"))

# Load Airline and Airport Data
with open('airlines.json', 'r') as f:
    AIRLINES = json.load(f)
with open('airports.json', 'r') as f:
    AIRPORTS = json.load(f)

# Load model results CSV
RESULTS_CSV = "model_summary_results.csv"
if os.path.exists(RESULTS_CSV):
    results_df = pd.read_csv(RESULTS_CSV)
else:
    results_df = pd.DataFrame()

# Available chart images from main.py
CHART_FILES = [
    "model_comparison.png",
    "confusion_matrices.png",
    "roc_curves.png",
    "precision_recall_curves.png",
    "feature_importance_comparison.png",
    "model_performance_radar.png",
    "cross_validation_analysis.png",
    "learning_curves.png",
    "calibration_curves.png",
    "cumulative_gain_chart.png",
    "lift_chart.png",
]


# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index-simplified.html", airlines=AIRLINES, airports=AIRPORTS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Parse date & time
        flight_date = datetime.strptime(data.get('date', '2026-04-15'), '%Y-%m-%d')
        flight_time = datetime.strptime(data.get('time', '12:00'), '%H:%M')

        # Build the 25-feature input matching FEATURE_NAMES from train_and_save.py
        input_dict = {
            'Month':                    flight_date.month,
            'Day_of_Week':              flight_date.weekday() + 1,
            'Departure_Hour':           flight_time.hour,
            'Departure_Minute':         flight_time.minute,
            'Distance_Miles':           float(data.get('distance', 500)),
            'Scheduled_Arrival_Time':   (flight_time.hour * 60 + flight_time.minute + int(float(data.get('distance', 500)) * 0.12)) % 1440,
            'Temperature_C':            float(data.get('temp', 22)),
            'Precipitation_mm':         float(data.get('precip', 0)),
            'Wind_Speed_kmh':           float(data.get('wind', 15)),
            'Visibility_km':            float(data.get('vis', 10)),
            'Humidity_Pct':             float(data.get('hum', 50)),
            'Pressure_hPa':             float(data.get('pres', 1013)),
            'Is_Weekend':               1 if flight_date.weekday() >= 5 else 0,
            'Is_Holiday':               0,
            'Aircraft_Age_Years':       float(data.get('age', 8)),
            'Num_Connections':          float(data.get('conns', 0)),
            'Previous_Delay_Risk':      0.15,
            'Airline_Encoded':          hash(data.get('airline', 'UA')) % 15,
            'Origin_Encoded':           hash(data.get('origin', 'SFO')) % 300,
            'Dest_Encoded':             hash(data.get('dest', 'LAX')) % 300,
            'Airport_Congestion_Index': float(data.get('congestion', 0.5)),
            'Passenger_Load_Factor':    float(data.get('load', 0.8)),
            'Fuel_Weight_kg':           float(data.get('fuel', 20000)),
            'Cargo_Weight_kg':          float(data.get('cargo', 5000)),
            'Ground_Crew_Efficiency':   float(data.get('efficiency', 0.9)),
        }

        # Build DataFrame in exact feature order
        df = pd.DataFrame([input_dict])
        X_ordered = df[feature_names]
        X_scaled = scaler.transform(X_ordered)

        # ── Prediction ──
        proba = float(xgb_model.predict_proba(X_scaled)[0][1])

        # ── XAI: SHAP values ──
        shap_values = explainer.shap_values(X_scaled)

        # Handle SHAP return shapes
        if isinstance(shap_values, list):
            # Binary classification returns [class0, class1]
            shap_vals = shap_values[1][0]
        elif shap_values.ndim == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # Top 5 contributing features
        feature_impacts = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanations = []
        for name, impact in feature_impacts[:5]:
            direction = "increased" if impact > 0 else "decreased"
            explanations.append({
                "feature": name.replace("_", " "),
                "impact": direction,
                "importance": round(abs(float(impact)), 4),
                "value": round(float(input_dict.get(name, 0)), 2)
            })

        # Determine risk level
        if proba > 0.8:
            risk = "CRITICAL"
        elif proba > 0.5:
            risk = "HIGH"
        elif proba > 0.3:
            risk = "MODERATE"
        else:
            risk = "LOW"

        return jsonify({
            "success": True,
            "prediction": "DELAYED" if proba > 0.5 else "ON TIME",
            "probability": round(proba, 4),
            "risk_level": risk,
            "explanations": explanations
        })

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        })


@app.route("/api/results")
def api_results():
    """Return model evaluation results as JSON."""
    if results_df.empty:
        return jsonify({"success": False, "error": "No results file found"})

    records = results_df.to_dict(orient="records")
    # Round floats
    for r in records:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 4)

    return jsonify({"success": True, "results": records})


@app.route("/api/charts")
def api_charts():
    """Return list of available chart filenames."""
    available = [f for f in CHART_FILES if os.path.exists(f)]
    return jsonify({"success": True, "charts": available})


@app.route("/api/charts/<filename>")
def serve_chart(filename):
    """Serve a chart image file."""
    filepath = os.path.join(os.getcwd(), filename)
    if os.path.exists(filepath) and filename.endswith('.png'):
        return send_file(filepath, mimetype='image/png')
    return jsonify({"error": "Chart not found"}), 404


# ─── Weather Data Function ─────────────────────────────────────────

AIRPORT_COORDS = {
    # Indian Airports
    "DEL": {"lat": 28.5665, "lon": 77.1031, "city": "New Delhi", "country": "India"},
    "BOM": {"lat": 19.0896, "lon": 72.8656, "city": "Mumbai", "country": "India"},
    "BLR": {"lat": 13.1939, "lon": 77.7068, "city": "Bangalore", "country": "India"},
    "HYD": {"lat": 17.3732, "lon": 78.4694, "city": "Hyderabad", "country": "India"},
    "COK": {"lat": 10.1591, "lon": 76.2192, "city": "Kochi", "country": "India"},
    "MAA": {"lat": 12.9940, "lon": 80.1689, "city": "Chennai", "country": "India"},
    "PNQ": {"lat": 18.5824, "lon": 73.9197, "city": "Pune", "country": "India"},
    "AMD": {"lat": 23.0225, "lon": 72.5714, "city": "Ahmedabad", "country": "India"},
    "SXR": {"lat": 34.2845, "lon": 75.5347, "city": "Srinagar", "country": "India"},
    "VTZ": {"lat": 17.9212, "lon": 83.3244, "city": "Visakhapatnam", "country": "India"},
    "LKO": {"lat": 26.7606, "lon": 80.8910, "city": "Lucknow", "country": "India"},
    "JAI": {"lat": 26.8124, "lon": 75.8028, "city": "Jaipur", "country": "India"},
    "VNS": {"lat": 25.3921, "lon": 82.8581, "city": "Varanasi", "country": "India"},
    "IXC": {"lat": 30.6733, "lon": 76.7850, "city": "Chandigarh", "country": "India"},
    "ATQ": {"lat": 31.7173, "lon": 74.8042, "city": "Amritsar", "country": "India"},
    "AGX": {"lat": 27.1445, "lon": 78.0092, "city": "Agra", "country": "India"},
    "IDR": {"lat": 22.7196, "lon": 75.8615, "city": "Indore", "country": "India"},
    "GOI": {"lat": 15.3810, "lon": 73.8344, "city": "Goa", "country": "India"},
    "COJ": {"lat": 11.1388, "lon": 75.9542, "city": "Kozhikode", "country": "India"},
    # US Airports
    "JFK": {"lat": 40.6413, "lon": -73.7781, "city": "New York", "country": "USA"},
    "LAX": {"lat": 33.9425, "lon": -118.4081, "city": "Los Angeles", "country": "USA"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "city": "Chicago", "country": "USA"},
    "DFW": {"lat": 32.8975, "lon": -97.0382, "city": "Dallas", "country": "USA"},
    "BOS": {"lat": 42.3656, "lon": -71.0096, "city": "Boston", "country": "USA"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "city": "Denver", "country": "USA"},
    "ATL": {"lat": 33.6407, "lon": -84.4277, "city": "Atlanta", "country": "USA"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "city": "San Francisco", "country": "USA"},
    "SEA": {"lat": 47.4502, "lon": -122.3088, "city": "Seattle", "country": "USA"},
    "MIA": {"lat": 25.7959, "lon": -80.2870, "city": "Miami", "country": "USA"},
    "LAS": {"lat": 36.0840, "lon": -115.1537, "city": "Las Vegas", "country": "USA"},
    "PHX": {"lat": 33.4373, "lon": -112.0078, "city": "Phoenix", "country": "USA"},
    "AUS": {"lat": 30.1975, "lon": -97.6664, "city": "Austin", "country": "USA"},
    "PHL": {"lat": 39.8744, "lon": -75.2424, "city": "Philadelphia", "country": "USA"},
    "MSP": {"lat": 44.8848, "lon": -93.2223, "city": "Minneapolis", "country": "USA"},
    "DCA": {"lat": 38.8512, "lon": -77.0402, "city": "Washington D.C.", "country": "USA"},
    "EWR": {"lat": 40.6895, "lon": -74.1745, "city": "Newark", "country": "USA"},
    "IAH": {"lat": 29.9902, "lon": -95.3368, "city": "Houston", "country": "USA"},
    "DTW": {"lat": 42.2124, "lon": -83.3534, "city": "Detroit", "country": "USA"},
    "IAD": {"lat": 38.9531, "lon": -77.4565, "city": "Washington Dulles", "country": "USA"},
    "SAN": {"lat": 32.7338, "lon": -117.1933, "city": "San Diego", "country": "USA"},
    "TPA": {"lat": 27.9756, "lon": -82.5333, "city": "Tampa", "country": "USA"},
    "MCO": {"lat": 28.4312, "lon": -81.3081, "city": "Orlando", "country": "USA"},
    "HNL": {"lat": 21.3187, "lon": -157.9225, "city": "Honolulu", "country": "USA"},
    "MCI": {"lat": 39.2976, "lon": -94.7139, "city": "Kansas City", "country": "USA"},
    "MDW": {"lat": 41.7868, "lon": -87.7522, "city": "Chicago Midway", "country": "USA"},
    "LGA": {"lat": 40.7772, "lon": -73.8726, "city": "New York LaGuardia", "country": "USA"},
    "SJC": {"lat": 37.3639, "lon": -121.9289, "city": "San Jose", "country": "USA"},
    "MSY": {"lat": 29.9934, "lon": -90.2580, "city": "New Orleans", "country": "USA"},
    "CLT": {"lat": 35.2140, "lon": -80.9431, "city": "Charlotte", "country": "USA"},
    "CVG": {"lat": 39.0488, "lon": -84.6678, "city": "Cincinnati", "country": "USA"},
    "SLC": {"lat": 40.7884, "lon": -111.9778, "city": "Salt Lake City", "country": "USA"},
    "PDX": {"lat": 45.5898, "lon": -122.5951, "city": "Portland", "country": "USA"},
    "STL": {"lat": 38.7487, "lon": -90.3700, "city": "St. Louis", "country": "USA"},
    "RDU": {"lat": 35.8801, "lon": -78.7880, "city": "Raleigh-Durham", "country": "USA"},
    "BNA": {"lat": 36.1263, "lon": -86.6774, "city": "Nashville", "country": "USA"},
    # International Airports
    "SIN": {"lat": 1.3644, "lon": 103.9915, "city": "Singapore", "country": "Singapore"},
    "BKK": {"lat": 13.6900, "lon": 100.7501, "city": "Bangkok", "country": "Thailand"},
    "DXB": {"lat": 25.2532, "lon": 55.3657, "city": "Dubai", "country": "UAE"},
    "DOH": {"lat": 25.2731, "lon": 51.6081, "city": "Doha", "country": "Qatar"},
    "KUL": {"lat": 2.7456, "lon": 101.7099, "city": "Kuala Lumpur", "country": "Malaysia"},
    "CGK": {"lat": -6.1256, "lon": 106.6559, "city": "Jakarta", "country": "Indonesia"},
    "HKG": {"lat": 22.3080, "lon": 113.9185, "city": "Hong Kong", "country": "China"},
    "TPE": {"lat": 25.0797, "lon": 121.2342, "city": "Taipei", "country": "Taiwan"},
    "ICN": {"lat": 37.4602, "lon": 126.4407, "city": "Seoul", "country": "South Korea"},
    "NRT": {"lat": 35.7720, "lon": 140.3929, "city": "Tokyo", "country": "Japan"},
    "LHR": {"lat": 51.4700, "lon": -0.4543, "city": "London", "country": "UK"},
    "CDG": {"lat": 49.0097, "lon": 2.5479, "city": "Paris", "country": "France"},
    "FRA": {"lat": 50.0379, "lon": 8.5622, "city": "Frankfurt", "country": "Germany"},
    "AMS": {"lat": 52.3105, "lon": 4.7683, "city": "Amsterdam", "country": "Netherlands"},
    "MUC": {"lat": 48.3537, "lon": 11.7750, "city": "Munich", "country": "Germany"},
}

def get_realistic_weather(airport_code, date_str):
    """Generate realistic weather data based on airport location and date."""
    try:
        if airport_code not in AIRPORT_COORDS:
            # Default weather if airport not found
            return generate_random_weather()
        
        coords = AIRPORT_COORDS[airport_code]
        flight_date = datetime.strptime(date_str, '%Y-%m-%d')
        month = flight_date.month
        
        # Base temperature by month and latitude
        lat = coords["lat"]
        if "India" in coords["country"]:
            # Indian climate
            base_temp_map = {1: 20, 2: 22, 3: 28, 4: 32, 5: 35, 6: 32, 7: 28, 8: 27, 9: 28, 10: 28, 11: 24, 12: 21}
            monsoon_months = [6, 7, 8, 9]
        else:
            # US climate
            base_temp_map = {1: 0, 2: 2, 3: 8, 4: 15, 5: 20, 6: 25, 7: 28, 8: 27, 9: 22, 10: 15, 11: 8, 12: 2}
            monsoon_months = []
        
        base_temp = base_temp_map.get(month, 20)
        temp_variation = random.randint(-5, 8)
        temperature = base_temp + temp_variation
        
        # Wind speed (higher during certain months)
        if month in [3, 4, 5, 6]:
            base_wind = random.randint(15, 30)
        else:
            base_wind = random.randint(8, 20)
        
        # Precipitation (higher during monsoon)
        if month in monsoon_months:
            precip = random.uniform(5, 20)
            visibility = random.uniform(5, 12)
            humidity = random.randint(75, 95)
        else:
            precip = random.uniform(0, 5)
            visibility = random.uniform(8, 15)
            humidity = random.randint(40, 70)
        
        return {
            "temperature": round(temperature, 1),
            "wind_speed": round(base_wind, 1),
            "precipitation": round(precip, 1),
            "visibility": round(visibility, 1),
            "humidity": humidity,
            "pressure": round(1013 + random.uniform(-5, 5), 1),
            "location": f"{coords['city']}, {coords['country']}"
        }
    except:
        return generate_random_weather()

def generate_random_weather():
    """Generate random but realistic weather data."""
    return {
        "temperature": round(random.uniform(10, 35), 1),
        "wind_speed": round(random.uniform(5, 40), 1),
        "precipitation": round(random.uniform(0, 15), 1),
        "visibility": round(random.uniform(5, 20), 1),
        "humidity": random.randint(20, 95),
        "pressure": round(1013 + random.uniform(-10, 10), 1),
        "location": "Unknown"
    }

@app.route("/api/weather", methods=["POST"])
def get_weather():
    """Get weather data for an airport on a specific date."""
    try:
        airport = request.json.get('airport')
        date = request.json.get('date', '2026-04-15')
        
        if not airport:
            return jsonify({"success": False, "error": "Airport code required"}), 400
        
        weather = get_realistic_weather(airport, date)
        return jsonify({
            "success": True,
            "airport": airport,
            "date": date,
            "weather": weather
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/fetch-all-weather", methods=["POST"])
def fetch_all_weather():
    """Fetch weather for both origin and destination airports."""
    try:
        origin = request.json.get('origin')
        dest = request.json.get('dest')
        date = request.json.get('date', '2026-04-15')
        
        origin_weather = get_realistic_weather(origin, date) if origin else None
        dest_weather = get_realistic_weather(dest, date) if dest else None
        
        return jsonify({
            "success": True,
            "origin": origin_weather,
            "destination": dest_weather
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ─── Haversine Distance Calculation ──────────────────────────────────
import math

def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth in miles."""
    R = 3958.8  # Earth's radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Airline Fleet Age Estimates ────────────────────────────────────
AIRLINE_FLEET_AGE = {
    "6E": 5, "AI": 14, "SG": 9, "UK": 4, "FL": 6, "I5": 7, "G8": 11,
    "UA": 16, "AA": 12, "DL": 17, "WN": 12, "B6": 11, "AS": 9, "NK": 6,
    "F9": 10, "HA": 18, "OO": 8, "EV": 13, "VX": 5, "MQ": 14, "US": 15,
    "SQ": 7, "EK": 8, "QR": 6, "BA": 13, "LH": 12, "AF": 11, "KL": 12,
    "QF": 14, "ET": 9, "TG": 15, "PK": 20, "BG": 18,
}

# ─── Airport Congestion Estimates ────────────────────────────────────
AIRPORT_CONGESTION = {
    # Major hubs = higher congestion
    "DEL": 0.82, "BOM": 0.80, "BLR": 0.65, "HYD": 0.55, "MAA": 0.60,
    "JFK": 0.88, "LAX": 0.85, "ORD": 0.87, "ATL": 0.90, "DFW": 0.78,
    "SFO": 0.75, "DEN": 0.72, "SEA": 0.65, "MIA": 0.70, "BOS": 0.72,
    "EWR": 0.82, "LGA": 0.85, "IAH": 0.68, "PHL": 0.70, "DCA": 0.73,
    "LHR": 0.92, "CDG": 0.80, "FRA": 0.78, "AMS": 0.75, "DXB": 0.85,
    "SIN": 0.70, "HKG": 0.80, "NRT": 0.75, "ICN": 0.72, "BKK": 0.68,
}


@app.route("/api/auto-fill", methods=["POST"])
def auto_fill():
    """Auto-fill all form parameters based on origin, destination, date, and airline."""
    try:
        data = request.json
        origin = data.get('origin', 'DEL')
        dest = data.get('dest', 'BOM')
        date_str = data.get('date', '2026-04-15')
        airline = data.get('airline', 'AI')
        time_str = data.get('time', '14:30')

        flight_date = datetime.strptime(date_str, '%Y-%m-%d')
        month = flight_date.month
        day_of_week = flight_date.weekday()  # 0=Mon, 6=Sun

        # ── Distance ──
        origin_coords = AIRPORT_COORDS.get(origin)
        dest_coords = AIRPORT_COORDS.get(dest)
        if origin_coords and dest_coords:
            distance = round(haversine_miles(
                origin_coords['lat'], origin_coords['lon'],
                dest_coords['lat'], dest_coords['lon']
            ))
        else:
            distance = 800  # fallback

        # ── Weather (origin airport - departure weather matters most) ──
        origin_weather = get_realistic_weather(origin, date_str)
        dest_weather = get_realistic_weather(dest, date_str)
        # Use the worse weather conditions (conservative estimate)
        temperature = origin_weather['temperature']
        wind_speed = max(origin_weather['wind_speed'], dest_weather['wind_speed'])
        precipitation = max(origin_weather['precipitation'], dest_weather['precipitation'])
        visibility = min(origin_weather['visibility'], dest_weather['visibility'])
        humidity = max(origin_weather['humidity'], dest_weather['humidity'])
        pressure = origin_weather['pressure']

        # ── Aircraft Age ──
        base_age = AIRLINE_FLEET_AGE.get(airline, 10)
        aircraft_age = base_age + random.randint(-2, 3)
        aircraft_age = max(1, min(35, aircraft_age))

        # ── Congestion (blend of both airports, weighted toward origin) ──
        origin_cong = AIRPORT_CONGESTION.get(origin, 0.50)
        dest_cong = AIRPORT_CONGESTION.get(dest, 0.50)
        congestion = round(origin_cong * 0.6 + dest_cong * 0.4, 2)
        # Time-of-day factor: higher congestion during peak hours
        try:
            hour = int(time_str.split(':')[0])
        except:
            hour = 14
        if 7 <= hour <= 10 or 17 <= hour <= 21:
            congestion = min(1.0, round(congestion + random.uniform(0.05, 0.12), 2))
        # Weekend effect
        if day_of_week >= 5:
            congestion = min(1.0, round(congestion + 0.05, 2))

        # ── Passenger Load Factor ──
        # Higher for popular routes & peak seasons
        base_load = 0.78
        if month in [6, 7, 12, 1]:  # peak travel months
            base_load = 0.88
        elif month in [3, 4, 10, 11]:  # shoulder season
            base_load = 0.82
        if day_of_week in [4, 6]:  # Fri & Sun popular
            base_load += 0.05
        load_factor = round(min(1.0, base_load + random.uniform(-0.05, 0.08)), 2)

        # ── Fuel Weight (based on distance) ──
        # ~3.5 kg per mile for short-haul, decreasing for long-haul
        if distance < 500:
            fuel = round(distance * 4.5 + random.randint(-500, 500))
        elif distance < 2000:
            fuel = round(distance * 3.8 + random.randint(-1000, 1000))
        else:
            fuel = round(distance * 3.2 + random.randint(-2000, 2000))
        fuel = max(5000, min(50000, fuel))

        # ── Cargo Weight (based on distance and aircraft size) ──
        if distance > 3000:  # long-haul = more cargo
            cargo = random.randint(8000, 18000)
        elif distance > 1000:
            cargo = random.randint(4000, 10000)
        else:
            cargo = random.randint(1500, 6000)

        # ── Ground Crew Efficiency ──
        # Major hubs tend to have better efficiency
        if origin_cong > 0.75:
            efficiency = round(random.uniform(0.82, 0.95), 2)
        else:
            efficiency = round(random.uniform(0.75, 0.98), 2)

        # ── Connections ──
        if distance > 4000:
            connections = random.choice([0, 1, 1])
        elif distance > 2000:
            connections = random.choice([0, 0, 1])
        else:
            connections = 0

        return jsonify({
            "success": True,
            "params": {
                "distance": distance,
                "temp": temperature,
                "wind": wind_speed,
                "precip": round(precipitation, 1),
                "vis": round(visibility, 1),
                "hum": humidity,
                "pres": round(pressure, 1),
                "age": aircraft_age,
                "congestion": congestion,
                "load": load_factor,
                "fuel": fuel,
                "cargo": cargo,
                "efficiency": efficiency,
                "conns": connections,
            },
            "weather_details": {
                "origin": {
                    "code": origin,
                    "location": origin_weather.get('location', origin),
                    "temp": origin_weather['temperature'],
                    "wind": origin_weather['wind_speed'],
                    "precip": origin_weather['precipitation'],
                    "vis": origin_weather['visibility'],
                    "hum": origin_weather['humidity'],
                },
                "dest": {
                    "code": dest,
                    "location": dest_weather.get('location', dest),
                    "temp": dest_weather['temperature'],
                    "wind": dest_weather['wind_speed'],
                    "precip": dest_weather['precipitation'],
                    "vis": dest_weather['visibility'],
                    "hum": dest_weather['humidity'],
                }
            },
            "route_info": {
                "origin_city": origin_coords['city'] if origin_coords else origin,
                "dest_city": dest_coords['city'] if dest_coords else dest,
                "distance_miles": distance,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "details": traceback.format_exc()}), 400


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AeroPredict AI — Flight Delay Prediction System")
    print("=" * 55)
    print(f"  Model features: {len(feature_names)}")
    print(f"  Airlines loaded: {len(AIRLINES)}")
    print(f"  Airports loaded: {len(AIRPORTS)}")
    print(f"  Charts available: {sum(1 for f in CHART_FILES if os.path.exists(f))}")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)
