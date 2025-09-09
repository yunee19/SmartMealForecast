import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from gensim.models import Word2Vec

# ===== Paths =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== Load dataset =====
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data_2_kcal.csv"), encoding='cp949')

# ===== Menu columns =====
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

# # ===== Feature Engineering =====
# feature_cols = ['Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp', 'Temp_Range',
#                 'Season', 'Month', 'Day', 'Actual_Emp', 'Total_Emp']
# ===== Feature Engineering =====
feature_cols = ['Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp', 'Temp_Range',
                'Season', 'Month', 'Day', 'Total_Emp']   # bỏ Actual_Emp

X = df[feature_cols].copy()
X['Season'] = X['Season'].astype('category').cat.codes

# ===== Pre_Special_Day =====
X['Pre_Special_Day'] = df['Special_Day'].shift(-1).fillna(0)

# # ===== Tỷ lệ nhân viên =====
# X['Emp_Ratio'] = df['Actual_Emp'] / df['Total_Emp']
# Train model dự đoán Actual_Emp
emp_features = ['Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day', 'Total_Emp']
X_emp = df[emp_features].copy()
X_emp['Season'] = X_emp['Season'].astype('category').cat.codes
y_emp = df['Actual_Emp']

dtrain_emp = xgb.DMatrix(X_emp, label=y_emp)
params_emp = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42, 'max_depth': 6, 'eta': 0.1}
model_emp = xgb.train(params_emp, dtrain_emp, num_boost_round=300)
df['Pred_Actual_Emp'] = model_emp.predict(xgb.DMatrix(X_emp))

# Tỷ lệ nhân viên dự đoán
X['Emp_Ratio'] = df['Pred_Actual_Emp'] / df['Total_Emp']

# ===== Sin/Cos encoding =====
X['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
X['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
X['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
X['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

# ===== WeekDay sin/cos =====
weekday_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4}
df['Week_Day_Num'] = df['Week_Day'].map(weekday_map)
X['WeekDay_sin'] = np.sin(2 * np.pi * df['Week_Day_Num'] / 5)
X['WeekDay_cos'] = np.cos(2 * np.pi * df['Week_Day_Num'] / 5)

# ===== Load Word2Vec =====
w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_menu.model"))

def get_menu_embedding(row, cols, model):
    vectors = [model.wv[str(row[col])] for col in cols if pd.notna(row[col]) and str(row[col]) in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

menu_embeddings = np.vstack(df.apply(lambda row: get_menu_embedding(row, menu_columns, w2v_model), axis=1))
menu_emb_df = pd.DataFrame(menu_embeddings, columns=[f"MenuVec_{i}" for i in range(menu_embeddings.shape[1])])

# ===== Combine features =====
X = pd.concat([X, menu_emb_df], axis=1)
X.columns = [str(col).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_") for col in X.columns]

# ===== Load models =====
model_lunch_xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_lunch_model_2.pkl"))
model_dinner_xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_dinner_model_2.pkl"))

# ===== Predict =====
df['Pred_Lunch_XGB'] = model_lunch_xgb.predict(xgb.DMatrix(X)).round().astype(int)
df['Pred_Dinner_XGB'] = model_dinner_xgb.predict(xgb.DMatrix(X)).round().astype(int)

# ===== Baseline =====
df['Baseline_Lunch'] = int(round(df['Lunch_Count'].mean()))
df['Baseline_Dinner'] = int(round(df['Dinner_Count'].mean()))

# ===== Save results =====
save_path = os.path.join(RESULTS_DIR, "predictions_all_data_2.csv")
df.to_csv(save_path, index=False)
print("Predictions saved to:", save_path)
