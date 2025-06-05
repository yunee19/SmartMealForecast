import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Step 1: Load the preprocessed dataset
# Bước 1: Tải dữ liệu đã tiền xử lý
X = pd.read_csv("X_encoded.csv")
y_lunch = pd.read_csv("y_lunch.csv").squeeze()  # Convert DataFrame to Series | Chuyển DataFrame thành Series
y_dinner = pd.read_csv("y_dinner.csv").squeeze()

# Step 2: Split the dataset into training and testing sets
# Bước 2: Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)

# Step 3: Train XGBoost models for lunch and dinner
# Bước 3: Huấn luyện mô hình XGBoost cho bữa trưa và bữa tối
model_lunch = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_lunch.fit(X_train, y_lunch_train)
y_lunch_pred = model_lunch.predict(X_test)

model_dinner = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_dinner.fit(X_train, y_dinner_train)
y_dinner_pred = model_dinner.predict(X_test)

# Step 4: Plot true vs predicted values for lunch
# Bước 4: Vẽ biểu đồ so sánh giá trị thực và dự đoán cho bữa trưa
plt.figure(figsize=(12, 5))
plt.plot(y_lunch_test.values[:100], label="Lunch True", marker='o')
plt.plot(y_lunch_pred[:100], label="Lunch Predicted", marker='x')
plt.title("Lunch - True vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Meal Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lineplot_lunch.png")
plt.show()

# Step 5: Plot true vs predicted values for dinner
# Bước 5: Vẽ biểu đồ so sánh giá trị thực và dự đoán cho bữa tối
plt.figure(figsize=(12, 5))
plt.plot(y_dinner_test.values[:100], label="Dinner True", marker='o')
plt.plot(y_dinner_pred[:100], label="Dinner Predicted", marker='x')
plt.title("Dinner - True vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Meal Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lineplot_dinner.png")
plt.show()

# Step 6: Create scatter plots to examine the relationship with temperature
# Bước 6: Vẽ biểu đồ phân tán để xem mối quan hệ với nhiệt độ

# Add predictions and actuals to a copy of X_test | Thêm cột dự đoán và thực tế vào X_test
X_test_with_temp = X_test.copy()
X_test_with_temp['Lunch_Pred'] = y_lunch_pred
X_test_with_temp['Dinner_Pred'] = y_dinner_pred
X_test_with_temp['Lunch_True'] = y_lunch_test.values
X_test_with_temp['Dinner_True'] = y_dinner_test.values

# Lunch scatter plot | Biểu đồ phân tán cho bữa trưa
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_test_with_temp, x="Avg_Temp", y="Lunch_True", label="Lunch Actual")
sns.scatterplot(data=X_test_with_temp, x="Avg_Temp", y="Lunch_Pred", label="Lunch Predicted")
plt.title("Avg Temp vs Lunch Count")
plt.xlabel("Average Temperature")
plt.ylabel("Meal Count")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_temp_lunch.png")
plt.show()

# Dinner scatter plot | Biểu đồ phân tán cho bữa tối
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_test_with_temp, x="Avg_Temp", y="Dinner_True", label="Dinner Actual")
sns.scatterplot(data=X_test_with_temp, x="Avg_Temp", y="Dinner_Pred", label="Dinner Predicted")
plt.title("Avg Temp vs Dinner Count")
plt.xlabel("Average Temperature")
plt.ylabel("Meal Count")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_temp_dinner.png")
plt.show()
