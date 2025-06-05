import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 📥 데이터 불러오기 함수 (Hàm tải dữ liệu)
def load_data():
    df = pd.read_csv("C:/Users/user/PycharmProjects/SmartMealForecast/data/processed/merged_data.csv", encoding='cp949')
    df['Date'] = pd.to_datetime(df['Date'])  # 문자열을 날짜로 변환 (Chuyển chuỗi thành kiểu ngày)
    df['Month'] = df['Date'].dt.month       # 월 정보 추출 (Lấy thông tin tháng)
    return df

# 📊 월별 메뉴 빈도 계산 함수 (Hàm tính tần suất món ăn theo từng tháng)
def get_monthly_trend(df):
    # 👉 Chỉ lấy món ăn chính: Lunch_Main hoặc Dinner_Main
    menu_columns = [col for col in df.columns if 'Lunch_Main' in col or 'Dinner_Main' in col]
    trend = {}

    for month in range(1, 13):
        menus = df[df['Month'] == month][menu_columns].values.flatten()
        menus = [m for m in menus if pd.notna(m)]  # 결측값 제거 (Loại bỏ giá trị NaN)
        count = Counter(menus)  # 메뉴별 빈도 계산 (Tính tần suất từng món)
        trend[month] = count

    return trend

# 🥇 월별로 가장 인기 있는 메뉴 출력 함수 (Hàm in món ăn được yêu thích nhất mỗi tháng)
def print_top_menu_each_month(trend):
    print("📅 월별로 가장 인기 있는 **메인** 메뉴 (Món ăn **chính** được yêu thích nhất mỗi tháng):")
    for month in range(1, 13):
        if trend[month]:
            top_menu = trend[month].most_common(1)[0]
            print(f"  {month}월 (Tháng {month}): {top_menu[0]} ({top_menu[1]}회 xuất hiện)")
        else:
            print(f"  {month}월 (Tháng {month}): 데이터 없음 (Không có dữ liệu)")

# 📈 월별 인기 메뉴 트렌드 시각화 함수 (Hàm vẽ xu hướng món ăn theo từng tháng)
def plot_top_menu_trend(trend, top_n=5):
    menu_freq = Counter()
    for counts in trend.values():
        menu_freq += counts

    # 전체 기간 중 가장 인기 있는 top N 메뉴 선택 (Chọn N món phổ biến nhất toàn bộ thời gian)
    top_menus = [m[0] for m in menu_freq.most_common(top_n)]

    for menu in top_menus:
        y = [trend[m].get(menu, 0) for m in range(1, 13)]  # 월별 등장 횟수 리스트 (Danh sách tần suất theo tháng)
        plt.plot(range(1, 13), y, label=menu)

    plt.xticks(range(1, 13))
    plt.xlabel("월 (Tháng)")
    plt.ylabel("출현 횟수 (Số lần xuất hiện)")
    plt.title("📈 월별 메인 메뉴 트렌드 (Xu hướng món chính theo tháng)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monthly_main_menu_trend.png")  # 결과를 이미지 파일로 저장 (Lưu kết quả thành file ảnh)
    print("✅ 저장 완료: monthly_main_menu_trend.png")  # Xác nhận đã lưu file
    plt.show()

# ▶️ 전체 실행 메인 함수 (Hàm chính)
def main():
    df = load_data()
    trend = get_monthly_trend(df)
    print_top_menu_each_month(trend)  # 월별 최고 인기 메인 메뉴 출력 (In món chính top từng tháng)
    plot_top_menu_trend(trend)        # 메인 메뉴 트렌드 그래프 출력 (Vẽ biểu đồ xu hướng món chính)

# 🟢 실행 시작 (Bắt đầu chạy)
if __name__ == "__main__":
    main()
