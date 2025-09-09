import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Cấu hình ---
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
BUILD_DIR = os.path.join(ROOT_DIR, "building_process")
DATA_DIR = os.path.join(BUILD_DIR, "data", "raw")
EDA_DIR = os.path.join(BUILD_DIR, "eda_results")

os.makedirs(EDA_DIR, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"), encoding="utf-8")

# Tổng số khách = Lunch + Dinner
df["Total_Count"] = df["Lunch_Count"] + df["Dinner_Count"]

# --- Tính weekday ---
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday  # 0=Mon, 6=Sun
df["Weekday"] = df["Date"].dt.day_name(locale="en_US")  # Mon, Tue,...
df["Weekday_num"] = df["Date"].dt.weekday  # 0=Mon ... 6=Sun

# Tính tỷ lệ khách theo thứ
weekday_ratio = df.groupby("Weekday_num")["Total_Count"].mean()
weekday_ratio = weekday_ratio / weekday_ratio.mean()
df["Weekday_Ratio"] = df["Weekday_num"].map(weekday_ratio)

# --- Biểu đồ khách trung bình theo thứ ---
plt.figure(figsize=(8, 5))
sns.barplot(x="Weekday_num", y="Total_Count", data=df, estimator=np.mean, palette="Blues_d")
plt.title("Average Customers by Weekday")
plt.xlabel("Weekday (0=Mon ... 6=Sun)")
plt.ylabel("Avg Total Customers")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "avg_customers_by_weekday.png"))
plt.close()

# --- Boxplot khách theo thứ ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="Weekday_num", y="Total_Count", data=df, palette="Set2")
plt.title("Distribution of Customers by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Total Customers")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "boxplot_customers_by_weekday.png"))
plt.close()

# --- Heatmap Holiday × Customer ---
pivot_holiday = df.pivot_table(values="Total_Count", index="Holiday", columns="Weekday_num", aggfunc="mean")
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_holiday, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap: Holiday × Weekday vs Customers")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "heatmap_holiday_weekday.png"))
plt.close()

# --- Save dataset với feature mới ---
df.to_csv(os.path.join(DATA_DIR, "merged_with_features.csv"), index=False, encoding="utf-8-sig")

print("✅ EDA completed. Charts saved to:", EDA_DIR)
