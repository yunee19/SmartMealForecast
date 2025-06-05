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
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== Load and preprocess data =====
data = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_all_data.csv'))
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Total_Customers'] = data['Lunch_Count'] + data['Dinner_Count']  # Tính cột tổng khách cho tất cả dữ liệu luôn
data['Special_Label'] = data['special_day'].apply(lambda x: '특별일' if x == 1 else '일반일')

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

def draw_boxplot_specialday(data):
    plt.figure(figsize=(8,6))
    # Sửa cảnh báo bằng cách thêm hue và legend=False
    sns.boxplot(x='Special_Label', y='Total_Customers', data=data, palette='pastel', hue='Special_Label', dodge=False, legend=False)
    sns.stripplot(x='Special_Label', y='Total_Customers', data=data, color='gray', alpha=0.4, jitter=True)
    plt.title('2020년 일반일 vs 특별일 고객 수 분포')
    plt.xlabel('날짜 유형')
    plt.ylabel('총 고객 수')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, '2020_boxplot_specialday_total_customers_improved.png'))
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
draw_boxplot_specialday(data_2020)
draw_scatter_by_month(data_2020)
draw_scatter_by_month_all_years(data)   # <--- Gọi hàm scatter plot tất cả năm
draw_barplot_season(data_2020)
draw_barplot_month(data_2020)
