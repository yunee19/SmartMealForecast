import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== Link =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")

os.makedirs(EVALUATION_DIR, exist_ok=True)

# ===== Load data =====
df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_all_data.csv"), encoding='utf-8')

# ===== real meals count =====
lunch_actual = df['Lunch_Count']
dinner_actual = df['Dinner_Count']
lunch_avg = lunch_actual.mean()
dinner_avg = dinner_actual.mean()

# ===== take predicted data =====
lunch_pred_cols = [col for col in df.columns if col.startswith('Lunch_Pred_')]
dinner_pred_cols = [col for col in df.columns if col.startswith('Dinner_Pred_')]

# =====  MAE and MAE Normalized =====
results = {
    "Model": [],
    "MAE_Lunch": [],
    "MAE_Dinner": [],
    "Normalized_MAE_Lunch": [],
    "Normalized_MAE_Dinner": []
}

for l_col, d_col in zip(lunch_pred_cols, dinner_pred_cols):
    model = l_col.replace("Lunch_Pred_", "")
    mae_lunch = np.mean(np.abs(lunch_actual - df[l_col]))
    mae_dinner = np.mean(np.abs(dinner_actual - df[d_col]))
    norm_lunch = mae_lunch / lunch_avg * 100
    norm_dinner = mae_dinner / dinner_avg * 100

    results["Model"].append(model)
    results["MAE_Lunch"].append(mae_lunch)
    results["MAE_Dinner"].append(mae_dinner)
    results["Normalized_MAE_Lunch"].append(norm_lunch)
    results["Normalized_MAE_Dinner"].append(norm_dinner)

# ===== save to CSV =====
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(EVALUATION_DIR, "mae_results.csv")
results_df.to_csv(results_csv_path, index=False)

# ===== graph =====
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
bar_width = 0.2
index = np.arange(len(results_df['Model']))

# ---- MAE ----
ax[0].bar(index, results_df['MAE_Lunch'], bar_width, label="Lunch MAE", color='skyblue', edgecolor='black')
ax[0].bar(index + bar_width, results_df['MAE_Dinner'], bar_width, label="Dinner MAE", color='orange', edgecolor='black')
ax[0].set_title("MAE Comparison (All Models)")
ax[0].set_xlabel("Model")
ax[0].set_ylabel("MAE")
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(results_df['Model'])
ax[0].legend()
for i, v in enumerate(results_df['MAE_Lunch']):
    ax[0].text(i, v + 2, f"{v:.2f}", ha='center')
for i, v in enumerate(results_df['MAE_Dinner']):
    ax[0].text(i + bar_width, v + 2, f"{v:.2f}", ha='center')

# ---- Normalized MAE ----
ax[1].bar(index, results_df['Normalized_MAE_Lunch'], bar_width, label="Lunch Norm MAE", color='skyblue', edgecolor='black')
ax[1].bar(index + bar_width, results_df['Normalized_MAE_Dinner'], bar_width, label="Dinner Norm MAE", color='orange', edgecolor='black')
ax[1].set_title("Normalized MAE Comparison (All Models)")
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Normalized MAE (%)")
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(results_df['Model'])
ax[1].legend()
for i, v in enumerate(results_df['Normalized_MAE_Lunch']):
    ax[1].text(i, v + 0.5, f"{v:.2f}%", ha='center')
for i, v in enumerate(results_df['Normalized_MAE_Dinner']):
    ax[1].text(i + bar_width, v + 0.5, f"{v:.2f}%", ha='center')

# ===== save graph =====
chart_path = os.path.join(EVALUATION_DIR, "mae_comparison_improved_chart.png")
plt.tight_layout()
plt.savefig(chart_path)
plt.show()

print(" Evaluation results saved to:", results_csv_path)
print(" Chart saved to:", chart_path)
