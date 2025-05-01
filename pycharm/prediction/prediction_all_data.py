import pandas as pd
import numpy as np
import joblib
from collections import Counter

# Load data
df = pd.read_csv("merged_data.csv", encoding='cp949')

# Define menu columns
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

# One-hot encoding
all_menus = pd.concat([df[col] for col in menu_columns])
unique_menus = all_menus.dropna().unique()
menu_ohe = pd.DataFrame(0, index=df.index, columns=unique_menus)

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_ohe.at[idx, val] = 1

# Feature Engineering
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X_all = df[feature_cols].copy()
X_all['Season'] = X_all['Season'].astype('category').cat.codes
X_all['Day'] = X_all['Day'].astype('category').cat.codes
X_all = pd.concat([X_all, menu_ohe], axis=1)
X_all.columns = X_all.columns.str.replace(r'[\[\]<>]', '_', regex=True)

# Load models
model_lunch_xgb = joblib.load("xgboost_lunch_model.pkl")
model_dinner_xgb = joblib.load("xgboost_dinner_model.pkl")
model_lunch_rf = joblib.load("ranfor_lunch_model.pkl")
model_dinner_rf = joblib.load("ranfor_dinner_model.pkl")

# Predict
df['Lunch_Pred_XGB'] = model_lunch_xgb.predict(X_all)
df['Dinner_Pred_XGB'] = model_dinner_xgb.predict(X_all)
df['Lunch_Pred_RF'] = model_lunch_rf.predict(X_all)
df['Dinner_Pred_RF'] = model_dinner_rf.predict(X_all)

# Baseline (mean of training target)
lunch_mean = df['Lunch_Count'].mean()
dinner_mean = df['Dinner_Count'].mean()
df['Lunch_Pred_Baseline'] = lunch_mean
df['Dinner_Pred_Baseline'] = dinner_mean

# Save results
df.to_csv("predictions_all_data.csv", index=False)
print("✅ Saved to predictions_all_data.csv")
