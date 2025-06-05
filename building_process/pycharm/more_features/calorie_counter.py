import pandas as pd

# Giả định dictionary calo
calorie_dict = {
    '밥': 300, '김치': 50, '된장국': 120, '불고기': 400, '계란찜': 150,
    '샐러드': 100, '두부조림': 180, '떡볶이': 350, '과일': 80, '물': 0
}

def calc_calories(menu_list):
    total = 0
    for item in menu_list:
        cal = calorie_dict.get(item, 0)
        print(f"  {item}: {cal} kcal")
        total += cal
    print(f" Total calo: {total} kcal")

def main():
    menu = input(" Type foods name (ex: 된장국,불고기): ")
    items = [i.strip() for i in menu.split(',')]
    calc_calories(items)

if __name__ == "__main__":
    main()
