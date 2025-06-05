import matplotlib.pyplot as plt
import seaborn as sns

results = {
    'XGBoost_Lunch': 68.42,
    'RandomForest_Lunch': 72.31,
    'Baseline_Lunch': 91.23,
    'LinearRegression': 77.85,
    'LightGBM': 70.14
}

plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("모델 MAE 성능 비교 / MAE Comparison")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("C:/Users/user/PycharmProjects/SmartMealForecast/pycharm/more_features/mae_comparison.png")
print("✅ MAE 비교 그래프 저장 완료: mae_comparison.png")
