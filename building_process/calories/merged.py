import pandas as pd

# --- 1. Đọc file CSV ---
menu_full = pd.read_csv("menu_full_with_all_kcal.csv", encoding='cp949', on_bad_lines='skip')
menu_fix = pd.read_csv("menu_fix.csv", encoding='cp949')

# --- 2. Cặp cột cần chèn calories ---
dish_col = 'Lunch_side_Dish_3'
kcal_col = 'Lunch_side_Dish_3_Kcal'

# --- 3. Tạo dictionary map món -> kcal ---
kcal_dict = dict(zip(menu_fix[dish_col], menu_fix[kcal_col]))

# --- 4. Chèn cột calories ngay cạnh cột món ---
col_index = menu_full.columns.get_loc(dish_col)
# Nếu cột kcal đã tồn tại, xóa trước khi chèn
if kcal_col in menu_full.columns:
    menu_full.drop(columns=[kcal_col], inplace=True)
menu_full.insert(col_index + 1, kcal_col, menu_full[dish_col].map(kcal_dict))

# --- 5. Xuất CSV với UTF-8 BOM ---
menu_full.to_csv("menu_full_with_Lunch_Side_Dish_3_Kcal.csv", index=False, encoding='utf-8-sig')

print(f"Đã chèn cột {kcal_col} cạnh {dish_col} và lưu file thành công!")
