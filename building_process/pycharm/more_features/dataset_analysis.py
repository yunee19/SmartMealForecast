import pandas as pd
import matplotlib.pyplot as plt

# 📌 데이터 불러오기 / Load dataset
# → merged_data.csv 파일에는 식수, 메뉴, 날씨 등의 데이터가 포함되어 있습니다
# → Đây là file chính chứa dữ liệu khách ăn, thực đơn, thời tiết...
df = pd.read_csv("C:/Users/user/PycharmProjects/SmartMealForecast/pycharm/prediction/merged_data.csv", encoding='cp949')

# ✅ 전체 행 개수 출력 / Print number of rows
print(f"👉 총 데이터 행 수: {len(df)}")  # Tổng số dòng dữ liệu

# ✅ 날짜 종류 수 출력 / Number of unique days
if 'Date' in df.columns:
    print(f"👉 날짜 종류 수: {df['Date'].nunique()}")  # Số ngày duy nhất
else:
    print("⚠️ 'Date' 컬럼이 없습니다. 날짜 정보가 필요합니다.")

# ✅ 점심/저녁 평균 인원 / Average Lunch & Dinner Count
print(f"👉 점심 평균 인원: {df['Lunch_Count'].mean():.2f}")  # Trung bình khách ăn trưa
print(f"👉 저녁 평균 인원: {df['Dinner_Count'].mean():.2f}")  # Trung bình khách ăn tối

# ✅ 메뉴 개수 분석 / Analyze number of unique menus
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi']

all_menus = pd.concat([df[col] for col in menu_columns])
menu_counts = all_menus.value_counts()

print(f"👉 고유 메뉴 개수: {len(menu_counts)}")  # Số lượng món ăn khác nhau

# ✅ 상위 인기 메뉴 시각화 / Top 20 popular menu visualization
menu_counts.head(20).plot(kind='barh', title='상위 20개 인기 메뉴')  # Top 20 menu phổ biến
plt.tight_layout()
plt.savefig("C:/Users/user/PycharmProjects/SmartMealForecast/pycharm/more_features/menu_distribution.png")
print("✅ 메뉴 분포 그래프 저장 완료 (menu_distribution.png)")