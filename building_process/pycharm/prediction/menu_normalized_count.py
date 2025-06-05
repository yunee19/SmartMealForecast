import pandas as pd
from collections import Counter

df = pd.read_csv("merged_data.csv", encoding='cp949')

menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

all_menus = pd.concat([df[col] for col in menu_columns]).dropna()
menu_counts = Counter(all_menus)
total = sum(menu_counts.values())

# Normalized count
menu_norm_df = pd.DataFrame(0, index=df.index, columns=menu_counts.keys())

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_norm_df.at[idx, val] = menu_counts[val] / total

# Save
menu_norm_df.to_csv("menu_normalized_count_encoded.csv", index=False)
print("✅ Normalized count encoding saved to menu_normalized_count_encoded.csv")
