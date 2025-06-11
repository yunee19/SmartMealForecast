import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 경로 설정 =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")

# 결과 저장할 폴더가 없다면 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== CSV 파일 불러오기 =====
csv_path = os.path.join(RESULTS_DIR, "predictions_all_data.csv")
df = pd.read_csv(csv_path)

# 날짜 형식 변환 및 정렬
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# 최근 30일 데이터만 추출
recent_df = df.tail(30)

# ===== 한글 폰트 설정 (운영체제에 따라 변경 필요) =====
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기준
plt.rcParams['axes.unicode_minus'] = False

# ===== 시각화 시작 =====

# ✅ 점심 예측 선 그래프 (최근 30일)
plt.figure(figsize=(15, 5))
plt.plot(recent_df['Date'], recent_df['Lunch_Count'], label='실제 점심 수량', color='black')
plt.plot(recent_df['Date'], recent_df['Lunch_Pred_XGB'], label='XGBoost 예측', linestyle='--')
plt.plot(recent_df['Date'], recent_df['Lunch_Pred_RF'], label='RandomForest 예측', linestyle='--')
plt.plot(recent_df['Date'], recent_df['Lunch_Pred_Baseline'], label='Baseline 예측', linestyle='--')
plt.title('최근 30일 점심 식수 예측 비교')
plt.xlabel('날짜')
plt.ylabel('식수량')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lunch_predictions_recent.png"))
plt.close()

# ✅ 저녁 예측 선 그래프 (최근 30일)
plt.figure(figsize=(15, 5))
plt.plot(recent_df['Date'], recent_df['Dinner_Count'], label='실제 저녁 수량', color='black')
plt.plot(recent_df['Date'], recent_df['Dinner_Pred_XGB'], label='XGBoost 예측', linestyle='--')
plt.plot(recent_df['Date'], recent_df['Dinner_Pred_RF'], label='RandomForest 예측', linestyle='--')
plt.plot(recent_df['Date'], recent_df['Dinner_Pred_Baseline'], label='Baseline 예측', linestyle='--')
plt.title('최근 30일 저녁 식수 예측 비교')
plt.xlabel('날짜')
plt.ylabel('식수량')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dinner_predictions_recent.png"))
plt.close()

# ✅ 산점도 함수 정의
def scatter_plot(actual_col, pred_col, title, filename):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=df[actual_col], y=df[pred_col])
    plt.plot([df[actual_col].min(), df[actual_col].max()],
             [df[actual_col].min(), df[actual_col].max()],
             'r--', label='y = x')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# ✅ 산점도 저장
scatter_plot('Lunch_Count', 'Lunch_Pred_XGB', '점심 - XGBoost 예측', "scatter_lunch_xgb.png")
scatter_plot('Dinner_Count', 'Dinner_Pred_XGB', '저녁 - XGBoost 예측', "scatter_dinner_xgb.png")
