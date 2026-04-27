# train_and_save.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import os

SEED = 42
np.random.seed(SEED)

# EXPANDED FEATURE LIST (25 features)
FEATURE_NAMES = [
    'Month', 'Day_of_Week', 'Departure_Hour', 'Departure_Minute',
    'Distance_Miles', 'Scheduled_Arrival_Time', 'Temperature_C', 
    'Precipitation_mm', 'Wind_Speed_kmh', 'Visibility_km',
    'Humidity_Pct', 'Pressure_hPa', 'Is_Weekend', 'Is_Holiday',
    'Aircraft_Age_Years', 'Num_Connections', 'Previous_Delay_Risk',
    'Airline_Encoded', 'Origin_Encoded', 'Dest_Encoded',
    'Airport_Congestion_Index', 'Passenger_Load_Factor', 'Fuel_Weight_kg',
    'Cargo_Weight_kg', 'Ground_Crew_Efficiency'
]

def generate_synthetic_data(n=30000):
    data = {
        'Month': np.random.randint(1, 13, n),
        'Day_of_Week': np.random.randint(1, 8, n),
        'Departure_Hour': np.random.randint(0, 24, n),
        'Departure_Minute': np.random.randint(0, 60, n),
        'Distance_Miles': np.random.randint(100, 3000, n),
        'Scheduled_Arrival_Time': np.random.randint(0, 1440, n),
        'Temperature_C': np.random.uniform(5, 35, n),
        'Precipitation_mm': np.random.exponential(scale=3, size=n).clip(0, 25), # Most days 0-5mm, extreme 20+
        'Wind_Speed_kmh': np.random.normal(loc=15, scale=8, size=n).clip(0, 50),
        'Visibility_km': np.random.uniform(3, 15, n),
        'Humidity_Pct': np.random.uniform(30, 95, n),
        'Pressure_hPa': np.random.uniform(1000, 1025, n),
        'Is_Weekend': np.random.randint(0, 2, n),
        'Is_Holiday': np.random.randint(0, 2, n),
        'Aircraft_Age_Years': np.random.randint(1, 25, n),
        'Num_Connections': np.random.randint(0, 3, n),
        'Previous_Delay_Risk': np.random.uniform(0, 0.5, n),
        'Airline_Encoded': np.random.randint(0, 15, n),
        'Origin_Encoded': np.random.randint(0, 300, n),
        'Dest_Encoded': np.random.randint(0, 300, n),
        'Airport_Congestion_Index': np.random.uniform(0.4, 0.95, n),
        'Passenger_Load_Factor': np.random.uniform(0.6, 0.95, n),
        'Fuel_Weight_kg': np.random.uniform(5000, 30000, n),
        'Cargo_Weight_kg': np.random.uniform(1000, 10000, n),
        'Ground_Crew_Efficiency': np.random.uniform(0.7, 0.99, n),
    }
    
    df = pd.DataFrame(data)
    
    # Complex delay logic for XAI to discover
    # Make the delay score highly sensitive to Congestion, Aircraft Age, and Load Factor
    # so that changing airlines and airports directly impacts the delay risk noticeably!
    score = (
        0.8 * df['Precipitation_mm'] + 
        0.4 * df['Wind_Speed_kmh'] - 
        0.6 * df['Visibility_km'] + 
        2.5 * df['Airport_Congestion_Index'] * 10 +
        1.5 * df['Passenger_Load_Factor'] * 10 +
        0.5 * df['Aircraft_Age_Years'] +
        0.1 * (df['Departure_Hour'] - 12)**2 / 5 +
        np.random.normal(0, 3, n)
    )
    df['DELAYED'] = (score > score.median()).astype(int)
    return df

# TRAIN
df = generate_synthetic_data()
X = df[FEATURE_NAMES]
y = df['DELAYED']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

xgb_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=SEED)
xgb_model.fit(X_scaled, y)

# XAI: SHAP Explainer
explainer = shap.TreeExplainer(xgb_model)

# SAVE
if not os.path.exists("models"): os.makedirs("models")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(FEATURE_NAMES, "models/feature_names.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")
xgb_model.save_model("models/xgb_model.json")

print("Advanced model and SHAP explainer saved.")
