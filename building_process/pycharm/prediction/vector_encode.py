import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load dataset
df = pd.read_csv("merged_data.csv", encoding='cp949')

# Define menu columns
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

merged_with_ohe = pd.concat([df, menu_ohe], axis=1)
merged_with_ohe.to_csv("One_hot_encoded.csv", index=False)

# ===== Word frequency encoding =====
menu_counter = Counter()
for col in menu_columns:
    menu_counter.update(df[col].dropna())

menu_word_encoded = pd.DataFrame(0, index=df.index, columns=unique_menus)
for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_word_encoded.at[idx, val] = menu_counter[val]

merged_with_word_encoding = pd.concat([df, menu_word_encoded], axis=1)
merged_with_word_encoding.to_csv("Word_encoded.csv", index=False)

# ===== Feature Engineering =====
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X = df[feature_cols].copy()
X['Season'] = X['Season'].astype('category').cat.codes
X['Day'] = X['Day'].astype('category').cat.codes
X = pd.concat([X, menu_ohe], axis=1)
X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)

y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']

# Train-test split
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)

# ===== Train Models =====
# --- XGBoost ---
model_lunch_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_dinner_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

model_lunch_xgb.fit(X_train, y_lunch_train)
model_dinner_xgb.fit(X_train, y_dinner_train)

y_lunch_pred_xgb = model_lunch_xgb.predict(X_test)
y_dinner_pred_xgb = model_dinner_xgb.predict(X_test)

# --- Random Forest ---
model_lunch_rf = RandomForestRegressor(random_state=42)
model_dinner_rf = RandomForestRegressor(random_state=42)

model_lunch_rf.fit(X_train, y_lunch_train)
model_dinner_rf.fit(X_train, y_dinner_train)

y_lunch_pred_rf = model_lunch_rf.predict(X_test)
y_dinner_pred_rf = model_dinner_rf.predict(X_test)

# ===== Evaluation =====
def evaluate(true, pred):
    return mean_squared_error(true, pred), mean_absolute_error(true, pred)

mse_lunch_xgb, mae_lunch_xgb = evaluate(y_lunch_test, y_lunch_pred_xgb)
mse_dinner_xgb, mae_dinner_xgb = evaluate(y_dinner_test, y_dinner_pred_xgb)

mse_lunch_rf, mae_lunch_rf = evaluate(y_lunch_test, y_lunch_pred_rf)
mse_dinner_rf, mae_dinner_rf = evaluate(y_dinner_test, y_dinner_pred_rf)

baseline_lunch = np.full_like(y_lunch_test, y_lunch_train.mean())
baseline_dinner = np.full_like(y_dinner_test, y_dinner_train.mean())
mse_lunch_base, mae_lunch_base = evaluate(y_lunch_test, baseline_lunch)
mse_dinner_base, mae_dinner_base = evaluate(y_dinner_test, baseline_dinner)

# ===== Save Evaluation =====
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("🔸 XGBoost Lunch\n")
    f.write(f"  MSE: {mse_lunch_xgb:.2f}, MAE: {mae_lunch_xgb:.2f}\n")
    f.write("🔸 XGBoost Dinner\n")
    f.write(f"  MSE: {mse_dinner_xgb:.2f}, MAE: {mae_dinner_xgb:.2f}\n\n")
    f.write("🔹 Random Forest Lunch\n")
    f.write(f"  MSE: {mse_lunch_rf:.2f}, MAE: {mae_lunch_rf:.2f}\n")
    f.write("🔹 Random Forest Dinner\n")
    f.write(f"  MSE: {mse_dinner_rf:.2f}, MAE: {mae_dinner_rf:.2f}\n\n")
    f.write("📌 Baseline Lunch\n")
    f.write(f"  MSE: {mse_lunch_base:.2f}, MAE: {mae_lunch_base:.2f}\n")
    f.write("📌 Baseline Dinner\n")
    f.write(f"  MSE: {mse_dinner_base:.2f}, MAE: {mae_dinner_base:.2f}\n")

# ===== Save Predictions =====
df_preds = df.copy()
df_preds['lunch_pred_xgb'] = np.nan
df_preds['dinner_pred_xgb'] = np.nan
df_preds['lunch_pred_rf'] = np.nan
df_preds['dinner_pred_rf'] = np.nan
df_preds['baseline_lunch_pred'] = np.nan
df_preds['baseline_dinner_pred'] = np.nan

df_preds.loc[X_test.index, 'lunch_pred_xgb'] = y_lunch_pred_xgb
df_preds.loc[X_test.index, 'dinner_pred_xgb'] = y_dinner_pred_xgb
df_preds.loc[X_test.index, 'lunch_pred_rf'] = y_lunch_pred_rf
df_preds.loc[X_test.index, 'dinner_pred_rf'] = y_dinner_pred_rf
df_preds.loc[X_test.index, 'baseline_lunch_pred'] = baseline_lunch
df_preds.loc[X_test.index, 'baseline_dinner_pred'] = baseline_dinner

df_preds.to_csv("predictions_all_models.csv", index=False)

# ===== Save Processed Data =====
X.to_csv("X_encoded.csv", index=False)
pd.DataFrame(y_lunch).to_csv("y_lunch.csv", index=False)
pd.DataFrame(y_dinner).to_csv("y_dinner.csv", index=False)

# ===== Save Models =====
joblib.dump(model_lunch_xgb, "xgboost_lunch_model.pkl")
joblib.dump(model_dinner_xgb, "xgboost_dinner_model.pkl")
joblib.dump(model_lunch_rf, "ranfor_lunch_model.pkl")
joblib.dump(model_dinner_rf, "ranfor_dinner_model.pkl")
# ===== Print Evaluation to Screen =====
print("🔸 XGBoost Lunch")
print(f"  MSE: {mse_lunch_xgb:.2f}, MAE: {mae_lunch_xgb:.2f}")
print("🔸 XGBoost Dinner")
print(f"  MSE: {mse_dinner_xgb:.2f}, MAE: {mae_dinner_xgb:.2f}\n")

print("🔹 Random Forest Lunch")
print(f"  MSE: {mse_lunch_rf:.2f}, MAE: {mae_lunch_rf:.2f}")
print("🔹 Random Forest Dinner")
print(f"  MSE: {mse_dinner_rf:.2f}, MAE: {mae_dinner_rf:.2f}\n")

print("📌 Baseline Lunch")
print(f"  MSE: {mse_lunch_base:.2f}, MAE: {mae_lunch_base:.2f}")
print("📌 Baseline Dinner")
print(f"  MSE: {mse_dinner_base:.2f}, MAE: {mae_dinner_base:.2f}")
