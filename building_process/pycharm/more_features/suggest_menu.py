import pandas as pd
from collections import Counter


# Load merged dataset
def load_data():
    df = pd.read_csv(r"C:\Users\user\PycharmProjects\SmartMealForecast\data\processed\merged_data.csv",
                     encoding='utf-8')
    return df


# Suggest most popular menus on busy days
def suggest_popular_menus(df, top_n=5):
    df = df[df['Lunch_Count'] > df['Lunch_Count'].mean()]  # Only use days with high lunch counts
    menu_columns = [col for col in df.columns if 'Lunch_' in col or 'Dinner_' in col]  # Menu-related columns
    all_menus = df[menu_columns].values.flatten()
    all_menus = [m for m in all_menus if pd.notna(m)]  # Remove NaN values
    most_common = Counter(all_menus).most_common(top_n)

    print("🍽️ Most Popular Menu Suggestions:")
    for menu, count in most_common:
        print(f"  {menu}: appeared {count} times")


# Main function
def main():
    df = load_data()
    suggest_popular_menus(df)


if __name__ == "__main__":
    main()
