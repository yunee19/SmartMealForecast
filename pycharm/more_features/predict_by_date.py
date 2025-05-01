import pandas as pd
import joblib
from datetime import datetime

# Get user input for date range
def get_user_dates():
    start = input("Enter start date (yyyy-mm-dd): ")
    end = input("Enter end date (yyyy-mm-dd): ")
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    return pd.date_range(start=start_date, end=end_date)

# Generate feature DataFrame based on dates
def create_feature_frame(dates):
    df = pd.DataFrame({'Date': dates})
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['Holiday'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # Saturday or Sunday = holiday
    df['Special_Day'] = 0  # Default: no special day
    df['Avg_Temp'] = 5  # Assumed temperature values
    df['Max_Temp'] = 10
    df['Min_Temp'] = 0
    df['Temp_Range'] = df['Max_Temp'] - df['Min_Temp']
    df['Season'] = df['Month'] % 12 // 3  # 0=Spring, 1=Summer, 2=Fall, 3=Winter
    return df

# Predict meal counts using trained models
def predict(df):
    model_dir = r"C:\Users\user\PycharmProjects\SmartMealForecast\pycharm\prediction"

    # Load pre-trained models
    rf_lunch = joblib.load(f"{model_dir}\\ranfor_lunch_model.pkl")
    rf_dinner = joblib.load(f"{model_dir}\\ranfor_dinner_model.pkl")
    xgb_lunch = joblib.load(f"{model_dir}\\xgboost_lunch_model.pkl")
    xgb_dinner = joblib.load(f"{model_dir}\\xgboost_dinner_model.pkl")

    # Define features to be used for prediction
    feature_cols = [
        'Holiday', 'Special_Day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
        'Temp_Range', 'Season', 'Month', 'DayOfWeek'
    ]

    # Generate predictions
    df['Lunch_RF'] = rf_lunch.predict(df[feature_cols])
    df['Dinner_RF'] = rf_dinner.predict(df[feature_cols])
    df['Lunch_XGB'] = xgb_lunch.predict(df[feature_cols])
    df['Dinner_XGB'] = xgb_dinner.predict(df[feature_cols])

    # Return result table
    return df[['Date', 'Lunch_RF', 'Dinner_RF', 'Lunch_XGB', 'Dinner_XGB']]

# Main execution function
def main():
    dates = get_user_dates()
    feature_df = create_feature_frame(dates)
    prediction_df = predict(feature_df)
    print(prediction_df)
    prediction_df.to_csv("predicted_custom_dates.csv", index=False)
    print("✅ Results saved to predicted_custom_dates.csv")

if __name__ == "__main__":
    main()
