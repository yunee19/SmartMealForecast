import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# ===== Paths =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== Load models =====
model_lunch = joblib.load(os.path.join(MODEL_DIR, "xgboost_lunch_model_2.pkl"))
model_dinner = joblib.load(os.path.join(MODEL_DIR, "xgboost_dinner_model_2.pkl"))

# ===== Load menu data =====
menu_df = pd.read_csv(os.path.join(DATA_DIR, "merged_data_2_kcal.csv"), encoding='cp949')

# ===== Input user information =====
print("Enter the date information for prediction:")
Year = int(input("Year (YYYY): "))
Special_Day = int(input("Special Day (0(No)/1(Yes)): "))
Avg_Temp = float(input("Average Temperature: "))
Max_Temp = float(input("Maximum Temperature: "))
Min_Temp = float(input("Minimum Temperature: "))
Temp_Range = float(input("Temperature Range: "))
Season = int(input("Season (0= Spring, 1= Summer, 2=Autumn , 3= Winter): "))
Month = int(input("Month (1-12): "))
Day = int(input("Day (1-31): "))
Total_Emp = int(input("Total Employees: "))
Week_Day = int(input("Week Day (0=Mon, 1=Tue, 2=Wed, 3= Thu, 4=Fri): "))

# ===== Emp_Ratio & Pre_Special_Day =====
Emp_Ratio = 0.8
Pre_Special_Day = 0

# ===== Sin/Cos encoding =====
Month_sin = np.sin(2 * np.pi * Month / 12)
Month_cos = np.cos(2 * np.pi * Month / 12)
Day_sin = np.sin(2 * np.pi * Day / 31)
Day_cos = np.cos(2 * np.pi * Day / 31)
WeekDay_sin = np.sin(2 * np.pi * Week_Day / 5)
WeekDay_cos = np.cos(2 * np.pi * Week_Day / 5)

# ===== Menu vectors =====
menu_vec = np.zeros(100)

# ===== Combine features =====
feature_names = ['Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp', 'Temp_Range',
                 'Season', 'Month', 'Day', 'Total_Emp', 'Pre_Special_Day', 'Emp_Ratio',
                 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'WeekDay_sin', 'WeekDay_cos']
feature_names += [f"MenuVec_{i}" for i in range(100)]

features = [Special_Day, Avg_Temp, Max_Temp, Min_Temp, Temp_Range,
            Season, Month, Day, Total_Emp, Pre_Special_Day, Emp_Ratio,
            Month_sin, Month_cos, Day_sin, Day_cos, WeekDay_sin, WeekDay_cos]
features.extend(menu_vec)

X_new = pd.DataFrame([features], columns=feature_names)

# ===== Predict servings =====
lunch_pred = model_lunch.predict(xgb.DMatrix(X_new)).round().astype(int)[0]
dinner_pred = model_dinner.predict(xgb.DMatrix(X_new)).round().astype(int)[0]

weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
Week_Day_str = weekday_map.get(Week_Day, str(Week_Day))

print(f"\nPredicted meals for {Year}/{Month}/{Day} ({Week_Day_str}):")
print(f"Lunch servings: {lunch_pred}, Dinner servings: {dinner_pred}")

# ===== Menu items =====
lunch_items = ['Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
               'Lunch_Side_Dish_2', 'Lunch_side_Dish_3', 'Lunch_Drink', 'Lunch_Kimchi']
dinner_items = ['Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
                'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi']

# ===== Function to pick random valid item =====
def random_menu_item(df, col_name):
    df_nonan = df[df[col_name+'_Kcal'].notna() & (df[col_name+'_Kcal']>0)]
    if df_nonan.empty:
        return "N/A", 0
    sample_row = df_nonan.sample(1).iloc[0]
    return sample_row[col_name], sample_row[col_name+'_Kcal']

# ===== Generate meal suggestions =====
lunch_suggestion = {item: random_menu_item(menu_df, item) for item in lunch_items}
dinner_suggestion = {item: random_menu_item(menu_df, item) for item in dinner_items}

# ===== Save to CSV =====
all_data = []
lunch_total = sum([kcal for name, kcal in lunch_suggestion.values()])
dinner_total = sum([kcal for name, kcal in dinner_suggestion.values()])

for name, (dish_name, kcal) in lunch_suggestion.items():
    all_data.append({'Meal':'Lunch','Predicted_Servings':lunch_pred,'Total_Kcal':lunch_total,
                     'Dish':dish_name,'Kcal':kcal})
for name, (dish_name, kcal) in dinner_suggestion.items():
    all_data.append({'Meal':'Dinner','Predicted_Servings':dinner_pred,'Total_Kcal':dinner_total,
                     'Dish':dish_name,'Kcal':kcal})

df_output = pd.DataFrame(all_data)
csv_path = os.path.join(RESULTS_DIR,f'meal_prediction_{Year}_{Month}_{Day}.csv')
df_output.to_csv(csv_path,index=False,encoding='utf-8-sig')
print(f"\nMeal suggestions saved to {csv_path}")

# ===== Print nicely =====
for meal_title, suggestion, total in [('Lunch', lunch_suggestion, lunch_total),
                                     ('Dinner', dinner_suggestion, dinner_total)]:
    print(f"\n{meal_title} Menu Suggestion:")
    print(f"{'Dish':<35} | {'Kcal':<5}")
    print("-"*45)
    for dish, (name, kcal) in suggestion.items():
        print(f"{name:<35} | {kcal:<5}Kcal")
    print("-"*45)
    print(f"{'Total':<35} | {total:<5}Kcal")
