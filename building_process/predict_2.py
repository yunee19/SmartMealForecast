import pandas as pd
import numpy as np
import joblib
import os

ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"))

# Feature Engineering (như trong train)
df["Date"] = pd.to_datetime(df[["Year","Month","Day"]])
df["Weekday_num"] = df["Date"].dt.weekday
df["sin_month"] = np.sin(2*np.pi*df["Month"]/12)
df["cos_month"] = np.cos(2*np.pi*df["Month"]/12)
df["sin_weekday"] = np.sin(2*np.pi*df["Weekday_num"]/7)
df["cos_weekday"] = np.cos(2*np.pi*df["Weekday_num"]/7)

def weather_category(temp):
    if temp < 0: return "Freezing"
    elif temp < 10: return "Cold"
    elif temp < 25: return "Mild"
    else: return "Hot"
df["Weather_Cat"] = df["Avg_Temp"].apply(weather_category).astype("category").cat.codes

feature_cols = [
    "Holiday","special_day","Avg_Temp","Max_Temp","Min_Temp","Temp_Range",
    "Season","Month","Day","Weekday_num",
    "Total_Emp","Actual_Emp","Leave_Emp","Trip_Emp","OT_Approved",
    "sin_month","cos_month","sin_weekday","cos_weekday","Weather_Cat"
]

X_all = df[feature_cols]

# Load models
rf_lunch = joblib.load(os.path.join(MODEL_DIR,"rf_lunch.pkl"))
rf_dinner = joblib.load(os.path.join(MODEL_DIR,"rf_dinner.pkl"))
xgb_lunch = joblib.load(os.path.join(MODEL_DIR,"xgb_lunch.pkl"))
xgb_dinner = joblib.load(os.path.join(MODEL_DIR,"xgb_dinner.pkl"))

# Predict
df["Lunch_Pred_RF"] = rf_lunch.predict(X_all).round().astype(int)
df["Dinner_Pred_RF"] = rf_dinner.predict(X_all).round().astype(int)
df["Lunch_Pred_XGB"] = xgb_lunch.predict(X_all).round().astype(int)
df["Dinner_Pred_XGB"] = xgb_dinner.predict(X_all).round().astype(int)

# Save
save_path = os.path.join(RESULTS_DIR,"predictions_v2.csv")
df.to_csv(save_path,index=False,encoding="utf-8-sig")
print("✅ Predictions saved to:", save_path)
