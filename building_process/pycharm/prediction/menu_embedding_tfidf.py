import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("merged_data.csv", encoding='cp949')

menu_cols = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]

# Combine menus as sentences
df['combined_menu'] = df[menu_cols].fillna('').agg(' '.join, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_menu'])

# Save result
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_csv("menu_tfidf_encoded.csv", index=False)
print("✅ TF-IDF menu vectorization saved to menu_tfidf_encoded.csv")
