import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from gensim.models import Word2Vec

# ===== Paths =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== Load dataset =====
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"), encoding='cp949')

# ===== Menu columns =====
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

# ===== Feature Engineering =====
feature_cols = ['Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp', 'Temp_Range', 'Season', 'Month', 'Day']
X = df[feature_cols].copy()
X['Season'] = X['Season'].astype('category').cat.codes

# --- Week_Day one-hot ---
X = pd.concat([X, pd.get_dummies(df['Week_Day'], prefix='WeekDay')], axis=1)

# --- Chu kỳ tuần (sin/cos) ---
week_map = {wd: i for i, wd in enumerate(df['Week_Day'].unique())}  # mapping 월,화,수... -> 0,1,2...
X['Week_Day_num'] = df['Week_Day'].map(week_map)
X['WeekDay_sin'] = np.sin(2 * np.pi * X['Week_Day_num'] / 7)
X['WeekDay_cos'] = np.cos(2 * np.pi * X['Week_Day_num'] / 7)
X.drop('Week_Day_num', axis=1, inplace=True)

# --- Tỷ lệ nhân viên làm việc ---
X['Emp_Ratio'] = df['Actual_Emp'] / df['Total_Emp']

# --- Sin/Cos Month & Day ---
X['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
X['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
X['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
X['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

# ===== Word2Vec Embedding for Menu =====
menu_sentences = [
    [str(row[col]) for col in menu_columns if pd.notna(row[col])]
    for idx, row in df.iterrows()
    if any(pd.notna(row[col]) for col in menu_columns)
]

w2v_model = Word2Vec(sentences=menu_sentences, vector_size=50, window=5, min_count=1, workers=4, seed=42)

def get_menu_embedding(row, cols, model):
    vectors = [model.wv[str(row[col])] for col in cols if pd.notna(row[col]) and str(row[col]) in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

menu_embeddings = np.vstack(df.apply(lambda row: get_menu_embedding(row, menu_columns, w2v_model), axis=1))
menu_emb_df = pd.DataFrame(menu_embeddings, columns=[f"MenuVec_{i}" for i in range(menu_embeddings.shape[1])])

X = pd.concat([X, menu_emb_df], axis=1)
X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)

# ===== Targets =====
y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']

# ===== Train-test split =====
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)

# ===== Train XGBoost Models =====
def build_xgb():
    return xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

model_lunch_xgb = build_xgb()
model_dinner_xgb = build_xgb()

model_lunch_xgb.fit(X_train, y_lunch_train)
model_dinner_xgb.fit(X_train, y_dinner_train)

# ===== Predict =====
y_lunch_pred = model_lunch_xgb.predict(X_test)
y_dinner_pred = model_dinner_xgb.predict(X_test)

# ===== Evaluation =====
def evaluate(true, pred):
    return mean_squared_error(true, pred), mean_absolute_error(true, pred)

mse_lunch, mae_lunch = evaluate(y_lunch_test, y_lunch_pred)
mse_dinner, mae_dinner = evaluate(y_dinner_test, y_dinner_pred)

baseline_lunch = np.full_like(y_lunch_test, y_lunch_train.mean())
baseline_dinner = np.full_like(y_dinner_test, y_dinner_train.mean())
mse_lunch_base, mae_lunch_base = evaluate(y_lunch_test, baseline_lunch)
mse_dinner_base, mae_dinner_base = evaluate(y_dinner_test, baseline_dinner)

# ===== Save Evaluation =====
with open(os.path.join(DATA_DIR, "evaluation_results_xgb.txt"), "w", encoding="utf-8") as f:
    f.write(" XGBoost Lunch\n")
    f.write(f"  MSE: {mse_lunch:.2f}, MAE: {mae_lunch:.2f}\n")
    f.write(" XGBoost Dinner\n")
    f.write(f"  MSE: {mse_dinner:.2f}, MAE: {mae_dinner:.2f}\n\n")
    f.write(" Baseline Lunch\n")
    f.write(f"  MSE: {mse_lunch_base:.2f}, MAE: {mae_lunch_base:.2f}\n")
    f.write(" Baseline Dinner\n")
    f.write(f"  MSE: {mse_dinner_base:.2f}, MAE: {mae_dinner_base:.2f}\n")

# ===== Save Models =====
joblib.dump(model_lunch_xgb, os.path.join(MODEL_DIR, "xgboost_lunch_model_xgb.pkl"))
joblib.dump(model_dinner_xgb, os.path.join(MODEL_DIR, "xgboost_dinner_model_xgb.pkl"))

print("XGBoost training completed! Evaluation results saved.")
