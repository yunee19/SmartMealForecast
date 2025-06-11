import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== Directory Setup =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
os.makedirs(EVALUATION_DIR, exist_ok=True)

# ===== Font settings (Korean) =====
plt.rcParams['font.family'] = 'Malgun Gothic'  # font tiếng Hàn
plt.rcParams['axes.unicode_minus'] = False     # để hiện dấu trừ

# ===== Load and preprocess data =====
data = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_all_data.csv'))
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Total_Customers'] = data['Lunch_Count'] + data['Dinner_Count']  # tổng khách trưa + tối
data['Special_Label'] = data['special_day'].apply(lambda x: '특별일' if x == 1 else '일반일')  # nhãn ngày đặc biệt

data_2020 = data[data['Date'].dt.year == 2020].copy()

# ===== Functions for visualization =====

def draw_boxplot_holiday(data):
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Holiday', y='Total_Customers', data=data)
    plt.title('휴일 여부에 따른 총 고객 수 분포 (2020년)')
    plt.xlabel('휴일 여부 (0: 평일, 1: 휴일)')
    plt.ylabel('총 고객 수')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, '2020_boxplot_holiday_total_customers.png'))
    plt.close()

def draw_scatter_by_month(data):
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle('2020년 월별 평균기온과 총 고객 수 관계', fontsize=16)
    for month in range(1, 13):
        ax = axes[(month-1)//3][(month-1)%3]
        monthly_data = data[data['Month'] == month]
        sns.scatterplot(x='Avg_Temp', y='Total_Customers', data=monthly_data, ax=ax)
        ax.set_title(f'{month}월')
        ax.set_xlabel('평균기온 (°C)')
        ax.set_ylabel('총 고객 수')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(EVALUATION_DIR, '2020_scatter_avgtemp_by_month.png'))
    plt.close()

def draw_scatter_by_month_all_years(data):
    if 'Year' not in data.columns:
        data['Year'] = data['Date'].dt.year

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle('월별 평균기온과 총 고객 수 관계 (모든 연도 데이터)', fontsize=16)
    for month in range(1, 13):
        ax = axes[(month-1)//3][(month-1)%3]
        monthly_data = data[data['Month'] == month]
        sns.scatterplot(x='Avg_Temp', y='Total_Customers', data=monthly_data, ax=ax)
        ax.set_title(f'{month}월')
        ax.set_xlabel('평균기온 (°C)')
        ax.set_ylabel('총 고객 수')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(EVALUATION_DIR, 'scatter_avgtemp_by_month_all_years.png'))
    plt.close()

def draw_barplot_season(data):
    plt.figure(figsize=(8,6))
    sns.barplot(x='Season', y='Total_Customers', data=data, estimator=sum)
    plt.title('계절별 총 고객 수 합계 (2020년)')
    plt.xlabel('계절')
    plt.ylabel('총 고객 수 합계')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, '2020_barplot_season_total_customers.png'))
    plt.close()

def draw_barplot_month(data):
    plt.figure(figsize=(10,6))
    sns.barplot(x='Month', y='Total_Customers', data=data, estimator=sum)
    plt.title('월별 총 고객 수 합계 (2020년)')
    plt.xlabel('월')
    plt.ylabel('총 고객 수 합계')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, '2020_barplot_month_total_customers.png'))
    plt.close()

# ===== Run All Plots =====
draw_boxplot_holiday(data_2020)
draw_scatter_by_month(data_2020)
draw_scatter_by_month_all_years(data)   # Gọi hàm scatter plot tất cả năm
draw_barplot_season(data_2020)
draw_barplot_month(data_2020)

# ===== Additional bar plot: Lunch_Count with special day labels =====
df = data.copy()
df['Date'] = pd.to_datetime(df['Date'])

# Chọn khoảng thời gian cụ thể
start_date = pd.to_datetime('2018-07-17')
end_date = pd.to_datetime('2018-08-17')
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

# Vẽ biểu đồ
plt.figure(figsize=(16, 6))
bars = plt.bar(df_filtered['Date'].dt.strftime('%Y-%m-%d'), df_filtered['Lunch_Count'],
               color='skyblue', edgecolor='black')

plt.xticks(rotation=45, ha='right')
plt.title("Lunch Customer Count from 2018-07-17 to 2018-08-17")
plt.xlabel("Date")
plt.ylabel("Number of Lunch Customers")

# Chú thích tên ngày đặc biệt nếu special_day == 1
for i, row in df_filtered.iterrows():
    if row['special_day'] == 1 and pd.notnull(row.get('Special_Day_Name')):
        plt.text(i - df_filtered.index[0], row['Lunch_Count'] + 2,
                 row['Special_Day_Name'],
                 ha='center', va='bottom', fontsize=9, color='red', rotation=45)

plt.tight_layout()

special_chart_path = os.path.join(EVALUATION_DIR, "barplot_special_day_lunch_count_customers.png")
plt.savefig(special_chart_path)
plt.show()

print("Barplot with special day names saved to:", special_chart_path)

# ===== Biểu đồ bổ sung: Mối quan hệ giữa nhiệt độ và số suất ăn trưa =====

# def draw_temp_vs_lunch_2019(data):
#     # Lọc dữ liệu tháng 2, 3, 4 năm 2019
#     df_2019_spring = data[
#         (data['Date'].dt.year == 2019) &
#         (data['Month'].isin([4,]))
#     ].copy()
#
#     # Tạo nhóm nhiệt độ
#     temp_bins = [-10, 0, 5, 10, 15, 20, 25, 30, 35]
#     temp_labels = ['<0°C', '0-5°C', '5-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '30°C+']
#     df_2019_spring['Temp_Group'] = pd.cut(df_2019_spring['Avg_Temp'], bins=temp_bins, labels=temp_labels)
#
#     # Tính trung bình suất ăn trưa theo từng nhóm nhiệt độ
#     grouped = df_2019_spring.groupby('Temp_Group')['Lunch_Count'].mean().reset_index()
#
#     # Vẽ biểu đồ
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Temp_Group', y='Lunch_Count', data=grouped, palette='YlOrRd')
#     plt.title('2019년 1~6월 평균기온 구간별 평균 점심 고객 수')
#     plt.xlabel('평균기온 구간')
#     plt.ylabel('평균 점심 고객 수')
#     plt.tight_layout()
#
#     save_path = os.path.join(EVALUATION_DIR, 'barplot_2019_temp_vs_lunch.png')
#     plt.savefig(save_path)
#     plt.show()
#     print("Biểu đồ được lưu tại:", save_path)
#
# # Gọi hàm:
# draw_temp_vs_lunch_2019(data)

def draw_temp_vs_lunch_2019(data):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Lọc dữ liệu tháng 4 năm 2019
    df_2019_spring = data[
        (data['Date'].dt.year == 2019) & (data['Month'] == 4)
    ].copy()

    # Bỏ các dòng có Avg_Temp bị thiếu
    df_2019_spring = df_2019_spring[df_2019_spring['Avg_Temp'].notna()]

    # Tạo nhóm nhiệt độ dựa trên cột Avg_Temp
    temp_bins = [-10, 0, 5, 10, 15, 20, 25, 30, 35]
    temp_labels = ['<0°C', '0-5°C', '5-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '30°C+']
    df_2019_spring['Temp_Group'] = pd.cut(df_2019_spring['Avg_Temp'], bins=temp_bins, labels=temp_labels)

    # Tính trung bình theo nhóm để vẽ line plot
    grouped = df_2019_spring.groupby('Temp_Group')['Lunch_Count'].mean().reset_index()

    # === Vẽ line plot ===
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Temp_Group', y='Lunch_Count', data=grouped, marker='o', sort=False)
    plt.title('📈 2019년 4월 평균기온 구간별 평균 점심 고객 수 (Line Plot)')
    plt.xlabel('평균기온 구간')
    plt.ylabel('평균 점심 고객 수')
    plt.tight_layout()
    save_path_line = os.path.join(EVALUATION_DIR, 'lineplot_2019_temp_vs_lunch.png')
    plt.savefig(save_path_line)
    plt.show()

    # === Vẽ box plot ===
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Temp_Group', y='Lunch_Count', data=df_2019_spring, palette='YlOrRd')
    plt.title('📦 2019년 4월 평균기온 구간별 점심 고객 수 분포 (Box Plot)')
    plt.xlabel('평균기온 구간')
    plt.ylabel('점심 고객 수')
    plt.tight_layout()
    save_path_box = os.path.join(EVALUATION_DIR, 'boxplot_2019_temp_vs_lunch.png')
    plt.savefig(save_path_box)
    plt.show()

draw_temp_vs_lunch_2019(data)