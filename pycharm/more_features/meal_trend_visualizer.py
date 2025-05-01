import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def load_data():
    df = pd.read_csv("merged_data.csv", encoding='cp949')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    return df

def get_monthly_trend(df):
    menu_columns = [col for col in df.columns if 'Lunch_' in col or 'Dinner_' in col]
    trend = {}

    for month in range(1, 13):
        menus = df[df['Month'] == month][menu_columns].values.flatten()
        menus = [m for m in menus if pd.notna(m)]
        count = Counter(menus)
        trend[month] = count

    return trend

def plot_top_menu_trend(trend, top_n=5):
    menu_freq = Counter()
    for counts in trend.values():
        menu_freq += counts

    top_menus = [m[0] for m in menu_freq.most_common(top_n)]
    for menu in top_menus:
        y = [trend[m].get(menu, 0) for m in range(1, 13)]
        plt.plot(range(1, 13), y, label=menu)

    plt.xticks(range(1, 13))
    plt.xlabel("Tháng")
    plt.ylabel("Tần suất xuất hiện")
    plt.title("📈 Xu hướng thực đơn theo tháng")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monthly_menu_trend.png")
    print("✅ Đã lưu biểu đồ vào monthly_menu_trend.png")
    plt.show()

def main():
    df = load_data()
    trend = get_monthly_trend(df)
    plot_top_menu_trend(trend)

if __name__ == "__main__":
    main()
