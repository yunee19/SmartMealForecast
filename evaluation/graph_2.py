import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ===== Directory Setup =====
ROOT_DIR = r"C:/Users/user/PycharmProjects/SmartMealForecast"
RESULTS_DIR = os.path.join(ROOT_DIR, "predictions_result")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")
os.makedirs(EVALUATION_DIR, exist_ok=True)

# ===== Font settings (Korean) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== Load predicted data =====
data = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_all_data_2.csv'))
data.columns = [col.strip() for col in data.columns]

# Convert date
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # dùng errors='coerce' để bỏ giá trị lỗi
data = data.dropna(subset=['Date'])  # loại bỏ các dòng Date lỗi
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Year'] = data['Date'].dt.year
data['Total_Customers'] = data['Pred_Lunch_XGB'] + data['Pred_Dinner_XGB']
data['Special_Label'] = data['Special_Day'].apply(lambda x: '특별일' if x == 1 else '일반일')


# ===== Boxplot by Special Day =====
def draw_boxplot_special_day(data):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Special_Label', y='Total_Customers', data=data)
    plt.title('Total Customers by Special Day Status (Predicted)')
    plt.xlabel('Special Day')
    plt.ylabel('Total Customers')
    plt.tight_layout()
    save_path = os.path.join(EVALUATION_DIR, 'boxplot_special_day_total_customers.png')
    plt.savefig(save_path)
    plt.show()
    print("Boxplot saved to:", save_path)


# ===== Scatter by Month =====
def draw_scatter_by_month(data):
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle('Monthly Avg Temp vs Total Customers', fontsize=16)
    for month in range(1, 13):
        ax = axes[(month - 1) // 3][(month - 1) % 3]
        monthly_data = data[data['Month'] == month]
        sns.scatterplot(x='Avg_Temp', y='Total_Customers', data=monthly_data, ax=ax)
        ax.set_title(f'{month}월')
        ax.set_xlabel('Avg Temp (°C)')
        ax.set_ylabel('Total Customers')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(EVALUATION_DIR, 'scatter_avgtemp_by_month.png')
    plt.savefig(save_path)
    plt.show()
    print("Scatter plot saved to:", save_path)


# ===== Barplot by Season =====
def draw_barplot_season(data):
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Season', y='Total_Customers', data=data, estimator=sum)
    plt.title('Total Customers by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Customers')
    plt.tight_layout()
    save_path = os.path.join(EVALUATION_DIR, 'barplot_season_total_customers.png')
    plt.savefig(save_path)
    plt.show()
    print("Barplot saved to:", save_path)


# ===== Line/Box Plot by Temp Groups =====
def draw_temp_vs_lunch_2019(data):
    df_2019_spring = data[(data['Date'].dt.year == 2019) & (data['Month'] == 4)].copy()
    df_2019_spring = df_2019_spring[df_2019_spring['Avg_Temp'].notna()]
    temp_bins = [-10, 0, 5, 10, 15, 20, 25, 30, 35]
    temp_labels = ['<0°C', '0-5°C', '5-10°C', '10-15°C', '15-20°C', '20-25°C', '25-30°C', '30°C+']
    df_2019_spring['Temp_Group'] = pd.cut(df_2019_spring['Avg_Temp'], bins=temp_bins, labels=temp_labels)

    grouped = df_2019_spring.groupby('Temp_Group')['Pred_Lunch_XGB'].mean().reset_index()

    # Line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Temp_Group', y='Pred_Lunch_XGB', data=grouped, marker='o', sort=False)
    plt.title('Avg Lunch Customers by Temp Group (April 2019, Predicted)')
    plt.xlabel('Avg Temp Group')
    plt.ylabel('Avg Lunch Customers')
    plt.tight_layout()
    save_path_line = os.path.join(EVALUATION_DIR, 'lineplot_2019_temp_vs_lunch.png')
    plt.savefig(save_path_line)
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Temp_Group', y='Pred_Lunch_XGB', data=df_2019_spring, palette='YlOrRd')
    plt.title('Lunch Customers Distribution by Temp Group (April 2019, Predicted)')
    plt.xlabel('Avg Temp Group')
    plt.ylabel('Lunch Customers')
    plt.tight_layout()
    save_path_box = os.path.join(EVALUATION_DIR, 'boxplot_2019_temp_vs_lunch.png')
    plt.savefig(save_path_box)
    plt.show()
    print("Line and Box plots saved:", save_path_line, save_path_box)


# ===== Menu Co-occurrence Heatmap =====
def menu_cooccurrence_heatmap(df, meal_type='Lunch', top_n=20):
    if meal_type == 'Lunch':
        menu_cols = ['Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
                     'Lunch_Side_Dish_2', 'Lunch_side_Dish_3']
    else:
        menu_cols = ['Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
                     'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3']
    menu_cols = [col for col in menu_cols if col in df.columns]
    all_items = pd.Series(df[menu_cols].values.ravel())
    top_items = all_items.value_counts().head(top_n).index.tolist()
    df_matrix = pd.DataFrame(0, index=df.index, columns=top_items)
    for col in menu_cols:
        for item in top_items:
            if col in df.columns:
                df_matrix[item] += df[col].eq(item).astype(int)
    co_occurrence = df_matrix.T.dot(df_matrix)
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'{meal_type} Menu Co-occurrence Heatmap (Top {top_n} items, Predicted)')
    plt.tight_layout()
    save_path = os.path.join(EVALUATION_DIR, f'{meal_type.lower()}_menu_cooccurrence_heatmap.png')
    plt.savefig(save_path)
    plt.show()
    print("Menu co-occurrence heatmap saved to:", save_path)


# ===== New Visualization: Special Day vs Total Customers =====
def heatmap_special_day_vs_customers(df):
    pivot = df.pivot_table(index='Special_Label', values='Total_Customers', aggfunc='mean')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Avg Total Customers by Special Day Status")
    plt.ylabel("Special Day")
    plt.xlabel("Avg Customers")
    save_path = os.path.join(EVALUATION_DIR, 'heatmap_special_day_total_customers.png')
    plt.savefig(save_path)
    plt.show()
    print("Heatmap saved to:", save_path)


# ===== New Visualization: Avg Customers per Main Dish =====
def avg_customers_per_main_dish(df, meal_type='Lunch', top_n=10):
    if meal_type == 'Lunch':
        main_col = 'Lunch_Main_Dish'
    else:
        main_col = 'Dinner_Main_Dish'

    top_dishes = df[main_col].value_counts().head(top_n).index.tolist()
    avg_customers = df[df[main_col].isin(top_dishes)].groupby(main_col)['Total_Customers'].mean().sort_values(
        ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_customers.values, y=avg_customers.index, palette='viridis')
    plt.title(f'Average Total Customers per {meal_type} Main Dish')
    plt.xlabel('Avg Total Customers')
    plt.ylabel('Main Dish')
    save_path = os.path.join(EVALUATION_DIR, f'avg_customers_per_{meal_type.lower()}_main_dish.png')
    plt.savefig(save_path)
    plt.show()
    print(f"{meal_type} Avg Customers per Main Dish saved to:", save_path)


# ===== Run all plots =====
draw_boxplot_special_day(data)
draw_scatter_by_month(data)
draw_barplot_season(data)
draw_temp_vs_lunch_2019(data)
menu_cooccurrence_heatmap(data, meal_type='Lunch', top_n=20)
menu_cooccurrence_heatmap(data, meal_type='Dinner', top_n=20)
heatmap_special_day_vs_customers(data)
avg_customers_per_main_dish(data, meal_type='Lunch', top_n=10)
avg_customers_per_main_dish(data, meal_type='Dinner', top_n=10)
