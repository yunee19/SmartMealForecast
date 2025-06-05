import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the predictions data
df = pd.read_csv("predictions_all_data.csv", encoding='utf-8')

# Calculate MAE for lunch and dinner
mae_lunch = np.mean(np.abs(df['Lunch_Count'] - df['Lunch_Pred_XGB']))  # MAE for lunch (XGB model)
mae_dinner = np.mean(np.abs(df['Dinner_Count'] - df['Dinner_Pred_XGB']))  # MAE for dinner (XGB model)

# Calculate MAE for lunch and dinner using the Random Forest model
mae_lunch_rf = np.mean(np.abs(df['Lunch_Count'] - df['Lunch_Pred_RF']))  # MAE for lunch (RF model)
mae_dinner_rf = np.mean(np.abs(df['Dinner_Count'] - df['Dinner_Pred_RF']))  # MAE for dinner (RF model)

# Calculate MAE for baseline
lunch_avg = df['Lunch_Count'].mean()
dinner_avg = df['Dinner_Count'].mean()
mae_lunch_baseline = np.mean(np.abs(df['Lunch_Count'] - lunch_avg))  # MAE for lunch (Baseline)
mae_dinner_baseline = np.mean(np.abs(df['Dinner_Count'] - dinner_avg))  # MAE for dinner (Baseline)

# Calculate MAE normalized by average count
mae_lunch_normalized = mae_lunch / lunch_avg * 100  # Normalized MAE for lunch
mae_dinner_normalized = mae_dinner / dinner_avg * 100  # Normalized MAE for dinner

mae_lunch_rf_normalized = mae_lunch_rf / lunch_avg * 100  # Normalized MAE for lunch (RF)
mae_dinner_rf_normalized = mae_dinner_rf / dinner_avg * 100  # Normalized MAE for dinner (RF)

mae_lunch_baseline_normalized = mae_lunch_baseline / lunch_avg * 100  # Normalized MAE for lunch (Baseline)
mae_dinner_baseline_normalized = mae_dinner_baseline / dinner_avg * 100  # Normalized MAE for dinner (Baseline)

# Prepare results for CSV file
results = {
    "Model": ["XGB", "RF", "Baseline"],
    "MAE_Lunch": [mae_lunch, mae_lunch_rf, mae_lunch_baseline],
    "MAE_Dinner": [mae_dinner, mae_dinner_rf, mae_dinner_baseline],
    "Normalized_MAE_Lunch": [mae_lunch_normalized, mae_lunch_rf_normalized, mae_lunch_baseline_normalized],
    "Normalized_MAE_Dinner": [mae_dinner_normalized, mae_dinner_rf_normalized, mae_dinner_baseline_normalized]
}

results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("mae_results.csv", index=False)

# Plot bar chart for MAE comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# MAE Bar Chart with better appearance
bar_width = 0.2
index = np.arange(len(results_df['Model']))

# Plotting MAE comparison (Lunch and Dinner)
ax[0].bar(index, results_df['MAE_Lunch'], bar_width, label="Lunch MAE", color='skyblue', edgecolor='black')
ax[0].bar(index + bar_width, results_df['MAE_Dinner'], bar_width, label="Dinner MAE", color='orange', edgecolor='black')
ax[0].set_title("MAE Comparison (XGB, RF, Baseline)")
ax[0].set_xlabel("Model")
ax[0].set_ylabel("MAE")
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(results_df['Model'])
ax[0].legend()

# Adding value labels on top of each bar
for i, v in enumerate(results_df['MAE_Lunch']):
    ax[0].text(i, v + 2, f"{v:.2f}", ha='center', va='bottom')
for i, v in enumerate(results_df['MAE_Dinner']):
    ax[0].text(i + bar_width, v + 2, f"{v:.2f}", ha='center', va='bottom')

# Normalized MAE Bar Chart with better appearance
ax[1].bar(index, results_df['Normalized_MAE_Lunch'], bar_width, label="Lunch Normalized MAE", color='skyblue', edgecolor='black')
ax[1].bar(index + bar_width, results_df['Normalized_MAE_Dinner'], bar_width, label="Dinner Normalized MAE", color='orange', edgecolor='black')
ax[1].set_title("Normalized MAE Comparison (XGB, RF, Baseline)")
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Normalized MAE (%)")
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(results_df['Model'])
ax[1].legend()

# Adding value labels on top of each bar
for i, v in enumerate(results_df['Normalized_MAE_Lunch']):
    ax[1].text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom')
for i, v in enumerate(results_df['Normalized_MAE_Dinner']):
    ax[1].text(i + bar_width, v + 0.5, f"{v:.2f}%", ha='center', va='bottom')

# Save the chart as an image
plt.tight_layout()
plt.savefig("mae_comparison_improved_chart.png")
plt.show()

print("✅ Kết quả đã được lưu vào 'mae_results.csv' và biểu đồ đã được lưu vào 'mae_comparison_improved_chart.png'")
