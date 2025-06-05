import pandas as pd
import datetime
import joblib  # Used to load pre-trained models / Dùng để load các mô hình đã được huấn luyện

# Load data from CSV file / Đọc dữ liệu từ file CSV
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime / Chuyển đổi cột 'Date' sang kiểu thời gian
    df['Month'] = df['Date'].dt.month       # Extract month / Trích xuất tháng
    df['Year'] = df['Date'].dt.year         # Extract year / Trích xuất năm
    return df

# Get training data for a specific month in previous years / Lấy dữ liệu huấn luyện cho một tháng trong các năm trước
def get_training_data(df, target_month, current_year):
    return df[(df['Month'] == target_month) & (df['Year'] < current_year)]

# Generate list of prediction dates for a specific month / Tạo danh sách ngày cần dự đoán trong một tháng cụ thể
def get_prediction_dates(year, month):
    start_date = datetime.date(year, month, 1)
    end_day = 28 if month == 2 else 30  # Simplify to 28 or 30 days / Giả sử tháng có 28 hoặc 30 ngày
    return pd.date_range(start=start_date, periods=end_day, freq='D')

# Prepare feature columns for model input / Chuẩn bị các đặc trưng đầu vào cho mô hình
def prepare_features(df):
    return df[['Month', 'Holiday', 'Avg_Temp', 'Max_Temp', 'Min_Temp']]

# Load saved models from files / Load các mô hình đã được lưu trước đó
def load_models():
    rf_lunch = joblib.load('ranfor_lunch_model.pkl')
    rf_dinner = joblib.load('ranfor_lunch_model.pkl')
    xgb_lunch = joblib.load('xgboost_lunch_model.pkl')
    xgb_dinner = joblib.load('xgboost_lunch_model.pkl')
    return rf_lunch, rf_dinner, xgb_lunch, xgb_dinner

# Predict lunch and dinner counts using saved models / Dự đoán số suất ăn trưa và tối bằng các mô hình đã lưu
def predict_with_saved_models(train_df, target_dates, original_df):
    y_lunch = train_df['Lunch_Count']
    y_dinner = train_df['Dinner_Count']

    rf_lunch, rf_dinner, xgb_lunch, xgb_dinner = load_models()

    # Dummy features for prediction / Đặc trưng giả định để dự đoán
    prediction_df = pd.DataFrame({
        'Month': [d.month for d in target_dates],
        'Holiday': [0]*len(target_dates),
        'Avg_Temp': [5]*len(target_dates),
        'Max_Temp': [10]*len(target_dates),
        'Min_Temp': [0]*len(target_dates)
    })

    # Baseline prediction using average / Dự đoán cơ sở bằng giá trị trung bình
    lunch_baseline = [int(y_lunch.mean())] * len(target_dates)
    dinner_baseline = [int(y_dinner.mean())] * len(target_dates)

    # Predictions from models / Dự đoán bằng các mô hình
    lunch_rf = rf_lunch.predict(prediction_df).astype(int)
    dinner_rf = rf_dinner.predict(prediction_df).astype(int)
    lunch_xgb = xgb_lunch.predict(prediction_df).astype(int)
    dinner_xgb = xgb_dinner.predict(prediction_df).astype(int)

    # Combine results / Kết hợp kết quả dự đoán
    result = pd.DataFrame({
        'Date': target_dates,
        'Lunch_Baseline': lunch_baseline,
        'Dinner_Baseline': dinner_baseline,
        'Lunch_RF': lunch_rf,
        'Dinner_RF': dinner_rf,
        'Lunch_XGB': lunch_xgb,
        'Dinner_XGB': dinner_xgb
    })

    # Merge with actual data if available / Ghép với dữ liệu thực tế nếu có
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    real_values = original_df[['Date', 'Lunch_Count', 'Dinner_Count']]
    result = result.merge(real_values, on='Date', how='left')

    return result

# Main function / Hàm chính
def main():
    file_path = 'predictions_all_data.csv'
    df = load_data(file_path)

    # User input / Nhập dữ liệu từ người dùng
    month_to_predict = int(input("Enter the month to predict (e.g., 8): "))
    current_year = int(input("Enter the year to predict (e.g., 2024): "))

    train_df = get_training_data(df, month_to_predict, current_year)
    target_dates = get_prediction_dates(current_year, month_to_predict)

    result = predict_with_saved_models(train_df, target_dates, df)

    # Show and save results / Hiển thị và lưu kết quả
    print("\nPrediction Results:")
    print(result)
    result.to_csv(f'prediction_month_{month_to_predict}_{current_year}.csv', index=False)
    print(f"Results saved to prediction_month_{month_to_predict}_{current_year}.csv")

if __name__ == "__main__":
    main()
