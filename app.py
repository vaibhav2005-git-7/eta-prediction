from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='.')

# Load models
BASE = os.path.dirname(os.path.abspath(__file__))
cls_model = joblib.load(os.path.join(BASE, 'models/xgboost_Classifier.pkl'))
reg_model = joblib.load(os.path.join(BASE, 'models/xgboost_regressor.pkl'))
cls_features = joblib.load(os.path.join(BASE, 'models/classification_feature_columns.pkl'))
reg_features = joblib.load(os.path.join(BASE, 'models/regression_feature_columns.pkl'))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    delivery_partner = data['delivery_partner']
    package_type = data['package_type']
    vehicle_type = data['vehicle_type']
    delivery_mode = data['delivery_mode']
    region = data['region']
    weather_condition = data['weather_condition']
    distance_km = float(data['distance_km'])
    package_weight_kg = float(data['package_weight_kg'])
    api_temperature = float(data['api_temperature'])
    api_humidity = float(data['api_humidity'])
    api_wind_speed = float(data['api_wind_speed'])
    holiday_or_weekend_transit_flag = int(data['holiday_or_weekend_transit_flag'])
    order_hour = int(data['order_hour'])
    order_day = int(data['order_day'])
    is_weekend = int(data['is_weekend'])

    bad_weather_flag = 1 if weather_condition in ['rainy', 'stormy'] else 0

    # --- Classification input ---
    cls_row = {
        'delivery_partner': delivery_partner,
        'package_type': package_type,
        'vehicle_type': vehicle_type,
        'delivery_mode': delivery_mode,
        'region': region,
        'weather_condition': weather_condition,
        'distance_km': distance_km,
        'package_weight_kg': package_weight_kg,
        'api_temperature': api_temperature,
        'api_humidity': api_humidity,
        'api_wind_speed': api_wind_speed,
        'bad_weather_flag_api': bad_weather_flag,
        'holiday_or_weekend_transit_flag': holiday_or_weekend_transit_flag,
        'order_hour': order_hour,
        'order_day': order_day,
        'is_weekend': is_weekend
    }

    cls_df = pd.DataFrame([cls_row])
    cls_df = pd.get_dummies(cls_df, drop_first=True)
    cls_df = cls_df.reindex(columns=cls_features, fill_value=0)

    delay_pred = int(cls_model.predict(cls_df)[0])
    delay_prob = float(cls_model.predict_proba(cls_df)[0][1])

    # --- Regression input ---
    reg_row = {
        'delivery_partner': delivery_partner,
        'package_type': package_type,
        'vehicle_type': vehicle_type,
        'delivery_mode': delivery_mode,
        'region': region,
        'weather_condition': weather_condition,
        'distance_km': distance_km,
        'package_weight_kg': package_weight_kg
    }

    reg_df = pd.DataFrame([reg_row])
    reg_df = pd.get_dummies(reg_df, drop_first=True)
    reg_df = reg_df.reindex(columns=reg_features, fill_value=0)

    eta_hours = float(reg_model.predict(reg_df)[0])

    return jsonify({
        'delayed': delay_pred,
        'delay_probability': round(delay_prob * 100, 1),
        'eta_hours': round(eta_hours, 2),
        'eta_minutes': round(eta_hours * 60, 0),
        'status': 'Likely Delayed' if delay_pred == 1 else 'On Time'
    })

if __name__ == '__main__':
    app.run()
