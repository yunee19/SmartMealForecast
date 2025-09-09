import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===== Paths =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
PRED_DIR = os.path.join(ROOT_DIR, "predictions_result")
os.makedirs(MODEL_DIR, exist_ok=True)

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

# ===== Word2Vec embedding =====
menu_sentences = []
for idx, row in df.iterrows():
    sentence = [str(row[col]) for col in menu_columns if pd.notna(row[col])]
    if sentence:
        menu_sentences.append(sentence)

w2v_model = Word2Vec(sentences=menu_sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)
w2v_model.save(os.path.join(MODEL_DIR, "w2v_menu.model"))

def get_menu_embedding(row, cols, model):
    vectors = [model.wv[str(row[col])] for col in cols if pd.notna(row[col]) and str(row[col]) in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

menu_embeddings = np.vstack(df.apply(lambda row: get_menu_embedding(row, menu_columns, w2v_model), axis=1))
menu_emb_df = pd.DataFrame(menu_embeddings, columns=[f"MenuVec_{i}" for i in range(menu_embeddings.shape[1])])

# ===== Combine features =====
X = pd.concat([X, menu_emb_df], axis=1)
X.columns = [str(col).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_") for col in X.columns]

# ===== Targets =====
y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']

# ===== Train-test split =====
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)

# Save feature names
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

# ===== Train XGBoost =====
def train_xgb(X_tr, y_tr, X_val, y_val):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
              'seed': 42, 'max_depth': 8, 'eta': 0.1}

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=evals, early_stopping_rounds=50, verbose_eval=50)
    return model

model_lunch_xgb = train_xgb(X_train, y_lunch_train, X_test, y_lunch_test)
model_dinner_xgb = train_xgb(X_train, y_dinner_train, X_test, y_dinner_test)

# ===== Predict =====
y_lunch_pred = model_lunch_xgb.predict(xgb.DMatrix(X_test))
y_dinner_pred = model_dinner_xgb.predict(xgb.DMatrix(X_test))

# ===== Evaluation =====
def evaluate(true, pred):
    return mean_squared_error(true, pred), mean_absolute_error(true, pred)

mse_lunch, mae_lunch = evaluate(y_lunch_test, y_lunch_pred)
mse_dinner, mae_dinner = evaluate(y_dinner_test, y_dinner_pred)

# ===== Baseline =====
baseline_lunch_pred = np.full_like(y_lunch_test, y_lunch_train.mean())
baseline_dinner_pred = np.full_like(y_dinner_test, y_dinner_train.mean())
mse_lunch_base, mae_lunch_base = evaluate(y_lunch_test, baseline_lunch_pred)
mse_dinner_base, mae_dinner_base = evaluate(y_dinner_test, baseline_dinner_pred)

# ===== Save evaluation =====
eval_df = pd.DataFrame({
    'Meal': ['Lunch', 'Dinner'],
    'MSE_Baseline': [mse_lunch_base, mse_dinner_base],
    'MAE_Baseline': [mae_lunch_base, mae_dinner_base],
    'MSE_XGBoost': [mse_lunch, mse_dinner],
    'MAE_XGBoost': [mae_lunch, mae_dinner]
})
eval_df.to_csv(os.path.join(EVALUATION_DIR, "train_2_evaluation.csv"), index=False)

# ===== Save predictions =====
pred_df = pd.DataFrame({
    'Actual_Lunch': y_lunch_test,
    'Pred_Lunch_XGB': y_lunch_pred,
    'Baseline_Lunch': baseline_lunch_pred,
    'Actual_Dinner': y_dinner_test,
    'Pred_Dinner_XGB': y_dinner_pred,
    'Baseline_Dinner': baseline_dinner_pred
})
pred_df.to_csv(os.path.join(PRED_DIR, "train_2_predictions.csv"), index=False)

# ===== Save models =====
joblib.dump(model_lunch_xgb, os.path.join(MODEL_DIR, "xgboost_lunch_model_2.pkl"))
joblib.dump(model_dinner_xgb, os.path.join(MODEL_DIR, "xgboost_dinner_model_2.pkl"))

# ===== Print evaluation =====
print("\n===== Evaluation Results =====")
print(f"{'Meal':<10} {'MSE_Base':>10} {'MAE_Base':>10} {'MSE_XGB':>10} {'MAE_XGB':>10}")
for idx, row in eval_df.iterrows():
    print(f"{row['Meal']:<10} {row['MSE_Baseline']:>10.2f} {row['MAE_Baseline']:>10.2f} {row['MSE_XGBoost']:>10.2f} {row['MAE_XGBoost']:>10.2f}")

print("Training completed! Models and predictions saved. Evaluation saved.")
