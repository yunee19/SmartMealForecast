import pandas as pd
import numpy as np
import joblib
import os
from collections import Counter

# ===== Link =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")


os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"))

# ===== Define menu columns =====
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

# ===== One-hot encoding =====
all_menus = pd.concat([df[col] for col in menu_columns])
unique_menus = all_menus.dropna().unique()
menu_ohe = pd.DataFrame(0, index=df.index, columns=unique_menus)

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_ohe.at[idx, val] = 1

# ===== Feature Engineering =====
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X_all = df[feature_cols].copy()
X_all['Season'] = X_all['Season'].astype('category').cat.codes
X_all['Day'] = X_all['Day'].astype('category').cat.codes
X_all = pd.concat([X_all, menu_ohe], axis=1)
X_all.columns = X_all.columns.str.replace(r'[\[\]<>]', '_', regex=True)

# ===== Load models =====
model_lunch_xgb = joblib.load(os.path.join(ROOT_DIR, "models", "xgboost_lunch_model.pkl"))
model_dinner_xgb = joblib.load(os.path.join(ROOT_DIR, "models", "xgboost_dinner_model.pkl"))
model_lunch_rf = joblib.load(os.path.join(ROOT_DIR, "models", "ranfor_lunch_model.pkl"))
model_dinner_rf = joblib.load(os.path.join(ROOT_DIR, "models", "ranfor_dinner_model.pkl"))

# ===== Predict (rounded as int) =====
df['Lunch_Pred_XGB'] = model_lunch_xgb.predict(X_all).round().astype(int)
df['Dinner_Pred_XGB'] = model_dinner_xgb.predict(X_all).round().astype(int)
df['Lunch_Pred_RF'] = model_lunch_rf.predict(X_all).round().astype(int)
df['Dinner_Pred_RF'] = model_dinner_rf.predict(X_all).round().astype(int)

# ===== Baseline (Mean rounded as int) =====
lunch_mean = int(round(df['Lunch_Count'].mean()))
dinner_mean = int(round(df['Dinner_Count'].mean()))
df['Lunch_Pred_Baseline'] = lunch_mean
df['Dinner_Pred_Baseline'] = dinner_mean

# ===== Save results =====
save_path = os.path.join(RESULTS_DIR, "predictions_all_data.csv")
df.to_csv(save_path, index=False)

print("Saved to :", save_path)
print(df[[ 'Lunch_Count', 'Lunch_Pred_XGB', 'Lunch_Pred_RF', 'Lunch_Pred_Baseline']].tail())


