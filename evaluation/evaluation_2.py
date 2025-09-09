import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== Paths =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
os.makedirs(EVALUATION_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_all_data_2.csv"), encoding='utf-8')

# ===== Real meals count =====
lunch_actual = df['Lunch_Count']
dinner_actual = df['Dinner_Count']
lunch_avg = lunch_actual.mean()
dinner_avg = dinner_actual.mean()

# ===== Predicted columns =====
lunch_pred_cols = ['Pred_Lunch_XGB', 'Baseline_Lunch']    # thêm các model nếu có sau này
dinner_pred_cols = ['Pred_Dinner_XGB', 'Baseline_Dinner']

# ===== Compute MAE and Normalized MAE =====
results = {
    "Model": [],
    "MAE_Lunch": [],
    "MAE_Dinner": [],
    "Normalized_MAE_Lunch": [],
    "Normalized_MAE_Dinner": []
}

for l_col, d_col in zip(lunch_pred_cols, dinner_pred_cols):
    model = l_col.replace("Pred_", "").replace("_Lunch", "").replace("_Dinner", "")
    mae_lunch = np.mean(np.abs(lunch_actual - df[l_col]))
    mae_dinner = np.mean(np.abs(dinner_actual - df[d_col]))
    norm_lunch = mae_lunch / lunch_avg * 100
    norm_dinner = mae_dinner / dinner_avg * 100

    results["Model"].append(model)
    results["MAE_Lunch"].append(mae_lunch)
    results["MAE_Dinner"].append(mae_dinner)
    results["Normalized_MAE_Lunch"].append(norm_lunch)
    results["Normalized_MAE_Dinner"].append(norm_dinner)

results_df = pd.DataFrame(results)

# ===== Save results to CSV =====
results_csv_path = os.path.join(EVALUATION_DIR, "mae_results_2_fixed.csv")
results_df.to_csv(results_csv_path, index=False)

# ===== Plot =====
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
bar_width = 0.25
index = np.arange(len(results_df))

# ---- MAE ----
ax[0].bar(index, results_df['MAE_Lunch'], bar_width, label="Lunch MAE", color='skyblue', edgecolor='black')
ax[0].bar(index + bar_width, results_df['MAE_Dinner'], bar_width, label="Dinner MAE", color='orange', edgecolor='black')
ax[0].set_title("MAE Comparison")
ax[0].set_xlabel("Model")
ax[0].set_ylabel("MAE")
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax[0].legend()
for i in range(len(results_df)):
    ax[0].text(i, results_df['MAE_Lunch'][i]+0.5, f"{results_df['MAE_Lunch'][i]:.2f}", ha='center')
    ax[0].text(i + bar_width, results_df['MAE_Dinner'][i]+0.5, f"{results_df['MAE_Dinner'][i]:.2f}", ha='center')

# ---- Normalized MAE ----
ax[1].bar(index, results_df['Normalized_MAE_Lunch'], bar_width, label="Lunch Norm MAE", color='skyblue', edgecolor='black')
ax[1].bar(index + bar_width, results_df['Normalized_MAE_Dinner'], bar_width, label="Dinner Norm MAE", color='orange', edgecolor='black')
ax[1].set_title("Normalized MAE Comparison")
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Normalized MAE (%)")
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax[1].legend()
for i in range(len(results_df)):
    ax[1].text(i, results_df['Normalized_MAE_Lunch'][i]+0.5, f"{results_df['Normalized_MAE_Lunch'][i]:.2f}%", ha='center')
    ax[1].text(i + bar_width, results_df['Normalized_MAE_Dinner'][i]+0.5, f"{results_df['Normalized_MAE_Dinner'][i]:.2f}%", ha='center')

# ===== Save and show chart =====
plt.tight_layout()
chart_path = os.path.join(EVALUATION_DIR, "mae_comparison_fixed_chart.png")
plt.savefig(chart_path)
plt.show()

