import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load dataset
df = pd.read_csv("merged_data.csv", encoding='cp949')  # hoặc utf-8 nếu cần

# Step 2: Define menu columns
menu_columns = [
    'Lunch_Rice','Lunch_Soup','Lunch_Main_Dish','Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2','Lunch_Drink','Lunch_Kimchi','Lunch_side_Dish_3',
    'Dinner_Rice','Dinner_Soup','Dinner_Main_Dish','Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2','Dinner_Side_Dish_3','Dinner_Drink','Dinner_Kimchi'
]

# Tạo word encoding (frequency)
menu_counter = Counter()
for col in menu_columns:
    menu_counter.update(df[col].dropna())

menu_word_encoded = pd.DataFrame(0, index=df.index, columns=menu_counter.keys())
for col in menu_columns:
    for idx, value in df[col].items():
        if pd.notna(value):
            menu_word_encoded.at[idx, value] = menu_counter[value]

# Merge word encoded vào df
merged_with_word_encoding = pd.concat([df, menu_word_encoded], axis=1)
merged_with_word_encoding.to_csv("Word_encoded.csv", index=False)

# Step 3: Feature set X and target y
feature_cols = [
    'Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
    'Temp_Range', 'Season', 'Month', 'Day'
]

X_word = df[feature_cols].copy()
X_word['Season'] = X_word['Season'].astype('category').cat.codes
X_word['Day'] = X_word['Day'].astype('category').cat.codes
X_word = pd.concat([X_word, menu_word_encoded], axis=1)
X_word.columns = X_word.columns.str.replace(r'[\[\]<>]', '_', regex=True)

y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']

# Step 4: Train-Test Split
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X_word, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X_word, y_dinner, test_size=0.2, random_state=42)

# Step 5: Train XGBoost Models
model_lunch = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_lunch.fit(X_train, y_lunch_train)

model_dinner = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_dinner.fit(X_train, y_dinner_train)

# Step 6: Predictions
y_lunch_pred = model_lunch.predict(X_test)
y_dinner_pred = model_dinner.predict(X_test)

# Step 7: Evaluation
mse_lunch = mean_squared_error(y_lunch_test, y_lunch_pred)
mae_lunch = mean_absolute_error(y_lunch_test, y_lunch_pred)
mse_dinner = mean_squared_error(y_dinner_test, y_dinner_pred)
mae_dinner = mean_absolute_error(y_dinner_test, y_dinner_pred)

print("XGBoost Lunch Prediction (Word Encoded)")
print(f"  MSE: {mse_lunch:.2f}")
print(f"  MAE: {mae_lunch:.2f}")
print("XGBoost Dinner Prediction (Word Encoded)")
print(f"  MSE: {mse_dinner:.2f}")
print(f"  MAE: {mae_dinner:.2f}")

# Step 8: Baseline
baseline_lunch = np.full_like(y_lunch_test, y_lunch_train.mean())
baseline_dinner = np.full_like(y_dinner_test, y_dinner_train.mean())

baseline_mse_lunch = mean_squared_error(y_lunch_test, baseline_lunch)
baseline_mae_lunch = mean_absolute_error(y_lunch_test, baseline_lunch)
baseline_mse_dinner = mean_squared_error(y_dinner_test, baseline_dinner)
baseline_mae_dinner = mean_absolute_error(y_dinner_test, baseline_dinner)

print("\nBaseline Lunch Prediction")
print(f"  MSE: {baseline_mse_lunch:.2f}")
print(f"  MAE: {baseline_mae_lunch:.2f}")
print("Baseline Dinner Prediction")
print(f"  MSE: {baseline_mse_dinner:.2f}")
print(f"  MAE: {baseline_mae_dinner:.2f}")

# Step 9: Random Forest
model_lunch_ranfor = RandomForestRegressor(random_state=42)
model_lunch_ranfor.fit(X_train, y_lunch_train)

model_dinner_ranfor = RandomForestRegressor(random_state=42)
model_dinner_ranfor.fit(X_train, y_dinner_train)

y_lunch_pred_ranfor = model_lunch_ranfor.predict(X_test)
y_dinner_pred_ranfor = model_dinner_ranfor.predict(X_test)

mse_lunch_ranfor = mean_squared_error(y_lunch_test, y_lunch_pred_ranfor)
mae_lunch_ranfor = mean_absolute_error(y_lunch_test, y_lunch_pred_ranfor)
mse_dinner_ranfor = mean_squared_error(y_dinner_test, y_dinner_pred_ranfor)
mae_dinner_ranfor = mean_absolute_error(y_dinner_test, y_dinner_pred_ranfor)

print("\n🔹 Random Forest Lunch Prediction (Word Encoded)")
print(f"  MSE: {mse_lunch_ranfor:.2f}")
print(f"  MAE: {mae_lunch_ranfor:.2f}")
print("🔹 Random Forest Dinner Prediction (Word Encoded)")
print(f"  MSE: {mse_dinner_ranfor:.2f}")
print(f"  MAE: {mae_dinner_ranfor:.2f}")

# Step 10: Save predictions
df_with_predictions = df.copy()
df_with_predictions['lunch_pred'] = np.nan
df_with_predictions['dinner_pred'] = np.nan
df_with_predictions['baseline_lunch_pred'] = np.nan
df_with_predictions['baseline_dinner_pred'] = np.nan
df_with_predictions['lunch_pred_ranfor'] = np.nan
df_with_predictions['dinner_pred_ranfor'] = np.nan

df_with_predictions.loc[X_test.index, 'lunch_pred'] = y_lunch_pred
df_with_predictions.loc[X_test.index, 'dinner_pred'] = y_dinner_pred
df_with_predictions.loc[X_test.index, 'baseline_lunch_pred'] = baseline_lunch
df_with_predictions.loc[X_test.index, 'baseline_dinner_pred'] = baseline_dinner
df_with_predictions.loc[X_test.index, 'lunch_pred_ranfor'] = y_lunch_pred_ranfor
df_with_predictions.loc[X_test.index, 'dinner_pred_ranfor'] = y_dinner_pred_ranfor

df_with_predictions.to_csv("prediction_data_word_encoded.csv", index=False)

# Step 11: Save results
with open("evaluation_results_word_encoded.txt", "w", encoding="utf-8") as f:
    f.write("XGBoost Lunch Prediction (Word Encoded)\n")
    f.write(f"  MSE: {mse_lunch:.2f}\n")
    f.write(f"  MAE: {mae_lunch:.2f}\n\n")
    f.write("XGBoost Dinner Prediction (Word Encoded)\n")
    f.write(f"  MSE: {mse_dinner:.2f}\n")
    f.write(f"  MAE: {mae_dinner:.2f}\n\n")
    f.write("Random Forest Lunch Prediction (Word Encoded)\n")
    f.write(f"  MSE: {mse_lunch_ranfor:.2f}\n")
    f.write(f"  MAE: {mae_lunch_ranfor:.2f}\n\n")
    f.write("Random Forest Dinner Prediction (Word Encoded)\n")
    f.write(f"  MSE: {mse_dinner_ranfor:.2f}\n")
    f.write(f"  MAE: {mae_dinner_ranfor:.2f}\n\n")
    f.write("Baseline Lunch Prediction\n")
    f.write(f"  MSE: {baseline_mse_lunch:.2f}\n")
    f.write(f"  MAE: {baseline_mae_lunch:.2f}\n\n")
    f.write("Baseline Dinner Prediction\n")
    f.write(f"  MSE: {baseline_mse_dinner:.2f}\n")
    f.write(f"  MAE: {baseline_mae_dinner:.2f}\n")

# Step 12: Save processed data and models
X_word.to_csv("X_word_encoded.csv", index=False)
pd.DataFrame(y_lunch).to_csv("y_lunch_word_encoded.csv", index=False)
pd.DataFrame(y_dinner).to_csv("y_dinner_word_encoded.csv", index=False)

joblib.dump(model_lunch, "xgboost_lunch_model_word_encoded.pkl")
joblib.dump(model_dinner, "xgboost_dinner_model_word_encoded.pkl")
joblib.dump(model_lunch_ranfor, "ranfor_lunch_model_word_encoded.pkl")
joblib.dump(model_dinner_ranfor, "ranfor_dinner_model_word_encoded.pkl")
