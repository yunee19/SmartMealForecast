import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:/Users/user/PycharmProjects/SmartMealForecast/pycharm/prediction/merged_data.csv", encoding='cp949')

columns_to_norm = ['Lunch_Count', 'Dinner_Count', 'Avg_Temp', 'Max_Temp', 'Min_Temp']
scaler = MinMaxScaler()
df[columns_to_norm] = scaler.fit_transform(df[columns_to_norm])

df.to_csv("C:/Users/user/PycharmProjects/SmartMealForecast/pycharm/more_features/merged_data_normalized.csv", index=False)
print("✅ 정규화 완료 및 저장됨: merged_data_normalized.csv")
