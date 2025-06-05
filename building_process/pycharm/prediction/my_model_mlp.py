import numpy as np

class SimpleLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Cập nhật weights và bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# ----- Sử dụng model mới -----

# Chuyển X_train, X_test sang numpy array
X_train_np = X_train.values
X_test_np = X_test.values

# Chuyển y sang numpy array
y_lunch_train_np = y_lunch_train.values
y_lunch_test_np = y_lunch_test.values

y_dinner_train_np = y_dinner_train.values
y_dinner_test_np = y_dinner_test.values

# Khởi tạo model tự code
model_lunch_custom = SimpleLinearRegression(lr=0.01, n_iters=2000)
model_dinner_custom = SimpleLinearRegression(lr=0.01, n_iters=2000)

# Huấn luyện
model_lunch_custom.fit(X_train_np, y_lunch_train_np)
model_dinner_custom.fit(X_train_np, y_dinner_train_np)

# Dự đoán
y_lunch_pred_custom = model_lunch_custom.predict(X_test_np)
y_dinner_pred_custom = model_dinner_custom.predict(X_test_np)

# Đánh giá
mse_lunch_custom, mae_lunch_custom = evaluate(y_lunch_test_np, y_lunch_pred_custom)
mse_dinner_custom, mae_dinner_custom = evaluate(y_dinner_test_np, y_dinner_pred_custom)

print(" Custom Linear Regression - Lunch")
print(f"  MSE: {mse_lunch_custom:.2f}, MAE: {mae_lunch_custom:.2f}")
print(" Custom Linear Regression - Dinner")
print(f"  MSE: {mse_dinner_custom:.2f}, MAE: {mae_dinner_custom:.2f}")

# Lưu dự đoán
df_preds['lunch_pred_custom'] = np.nan
df_preds['dinner_pred_custom'] = np.nan
df_preds.loc[X_test.index, 'lunch_pred_custom'] = y_lunch_pred_custom
df_preds.loc[X_test.index, 'dinner_pred_custom'] = y_dinner_pred_custom

df_preds.to_csv(os.path.join(DATA_DIR, "predictions_train_result.csv"), index=False)
