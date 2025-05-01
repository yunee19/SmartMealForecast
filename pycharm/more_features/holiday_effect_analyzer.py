import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("merged_data.csv", encoding='cp949')
    return df

def analyze(df):
    df['is_holiday'] = df['Holiday'].apply(lambda x: 'Holiday' if x == 1 else 'Normal')
    lunch_avg = df.groupby('is_holiday')['Lunch_Count'].mean()
    dinner_avg = df.groupby('is_holiday')['Dinner_Count'].mean()

    print("🍱 Trung bình khách ăn trưa:")
    print(lunch_avg)
    print("🍛 Trung bình khách ăn tối:")
    print(dinner_avg)

    sns.boxplot(data=df, x='is_holiday', y='Lunch_Count')
    plt.title("So sánh lượng khách ăn trưa (Ngày lễ vs Bình thường)")
    plt.savefig("lunch_holiday_compare.png")
    plt.show()

    sns.boxplot(data=df, x='is_holiday', y='Dinner_Count')
    plt.title("So sánh lượng khách ăn tối (Ngày lễ vs Bình thường)")
    plt.savefig("dinner_holiday_compare.png")
    plt.show()

def main():
    df = load_data()
    analyze(df)

if __name__ == "__main__":
    main()
