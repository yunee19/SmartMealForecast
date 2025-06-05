
# SmartMealForecast 프로젝트 README

## 1. 프로젝트 소개  
작품명: SmartMealForecast (식수 예측 시스템)  
SmartMealForecast는 하루 동안 필요한 고객 수 또는 식수량을 예측하는 시스템을 구축하는 프로젝트입니다.  

**주요 고려 요소**  
- 과거 고객 및 식수량 데이터: 반복되는 패턴을 학습하여 예측 정확도를 높임  
- 날씨 데이터: 외출 여부와 식사 선택에 큰 영향  
- 공휴일 및 특별 이벤트: 고객 수요의 증가 요인이 될 수 있음  

**예측의 중요성**  
정확한 예측은 음식 낭비를 줄이고, 부족하거나 과잉되는 상황을 방지하며, 고객 만족도를 향상시킵니다.  

**사용 알고리즘**  
XGBoost와 Random Forest와 같은 머신러닝 알고리즘을 활용하여 비선형적이고 복잡한 데이터를 효과적으로 처리합니다.  

---

## 2. 프로젝트 구조  
```

SmartMealForecast/
├── data/                        ← 원본 데이터 및 전처리 데이터
│   ├── merged\_data.csv          ← 전처리 완료 데이터 (컬럼명 변경, 특수일 데이터 처리 등)
│   ├── one\_hot\_encoded.csv      ← 원-핫 인코딩 데이터
│   ├── word\_encoded.csv         ← 단어 빈도 인코딩 데이터
│   ├── predictions\_train\_result.csv ← 훈련 결과 데이터
│   ├── evaluation\_results.txt   ← XGBoost, Random Forest, Baseline의 MSE 및 MAE 결과
│   ├── x\_encoded.csv            ← 특성 공학 데이터
│   ├── y\_lunch.csv              ← 점심 고객 수 데이터
│   ├── y\_dinner.csv             ← 저녁 고객 수 데이터
│   ├── original\_data.csv        ← 원본 데이터
│   ├── original\_meal\_data\_2016\_2021.csv ← 2016\~2021 식사 데이터 원본
│   ├── original\_special\_day\_data\_2016\_2021.csv ← 휴일 정보 원본
│   └── original\_weather\_data\_2016\_2021.csv ← 날씨 데이터 원본

├── prediction/                 ← 예측 수행 관련 스크립트
│   ├── train.py                ← 인코딩, 모델 훈련 및 결과 통합
│   ├── prediction.py           ← 전체 데이터에 대한 예측 실행
│   └── my\_model.py             ← 모델 정의 코드

├── models/                     ← 학습된 모델 저장
│   ├── ranfor\_dinner\_model.pkl
│   ├── ranfor\_lunch\_model.pkl
│   ├── xgboost\_dinner\_model.pkl
│   └── xgboost\_model.pkl

├── evaluation/                 ← 평가 결과 및 시각화 자료
│   ├── evaluation.py           ← MAE, Normalized MAE 계산 및 그래프 생성 스크립트
│   ├── graph.py                ← 다양한 시각화 그래프 생성 스크립트
│   ├── mae\_comparison\_improved\_chart.png ← MAE 비교 그래프
│   ├── mae\_result.csv          ← 평가 결과 CSV 파일
│   ├── 2020\_boxplot\_holiday\_total\_customers.png
│   ├── 2020\_boxplot\_specialday\_total\_customers\_improved.png
│   ├── 2020\_scatter\_avgtemp\_by\_month.png
│   ├── scatter\_avgtemp\_by\_month\_all\_years.png
│   ├── 2020\_barplot\_season\_total\_customers.png
│   └── 2020\_barplot\_month\_total\_customers.png

├── prediction\_result/          ← 예측 결과 저장
│   └── predictions\_all\_data.csv

└── README.md                   ← 프로젝트 설명서 (현재 파일)

````
---

## 3. 사용한 기술 및 라이브러리  
- Python 3.x  
- Pandas, NumPy (데이터 처리)  
- Scikit-learn (머신러닝 모델)  
- XGBoost (부스팅 모델)  
- Matplotlib, Seaborn (데이터 시각화)  

---

## 4. 주요 기능 및 역할  
- 데이터 전처리 및 인코딩 (One-hot, 단어 빈도 등)  
- 점심 및 저녁 고객 수 예측 모델 훈련 및 예측  
- 다양한 머신러닝 알고리즘 적용 (XGBoost, Random Forest 등)  
- 예측 결과 평가 (MAE, Normalized MAE, MSE)  
- 결과 시각화 (박스플롯, 산점도, 바플롯 등)  

---

## 5. 실행 방법  
1. 필요한 라이브러리 설치  
```bash
pip install -r requirements.txt
````

2. 데이터 준비: `data/` 폴더에 원본 데이터 저장
3. 모델 훈련 및 예측 실행

```bash
python prediction/train.py  
python prediction/prediction.py
```

4. 평가 및 시각화 실행

```bash
python evaluation/evaluation.py  
python evaluation/graph.py
```

---

## 6. 데이터 설명

* 총 1205행 데이터, 출처는 Dacon (데이터 고객/주문 : [Dacon 링크](https://dacon.io/competitions/official/235743/overview/description?utm_source=chatgpt.com))
* 날씨 데이터 출처 : [기상청 데이터](https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179)
* 휴일 및 특별일 데이터는 달력에서 수집
* 엑셀을 이용해 분석하기 쉽게 데이터를 분리하고 필요한 원본 데이터를 병합하여 `merged_data.csv` 파일 생성

---

## 7. train.py 파일 상세 분석

**train.py 주요 목적**

* 준비된 데이터(`merged_data.csv`) 읽기
* 머신러닝 모델 학습을 위한 특성(feature) 생성
* 점심 및 저녁 고객 수 예측 모델 학습 (Lunch\_Count, Dinner\_Count)
* 평가 및 결과 저장

---

**train.py 상세 단계**

1. 병합 데이터 읽기 (`merged_data.csv`)

```python
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"))
```

* 입력 데이터는 날씨 정보, 특별한 날, 음식 메뉴 등 다양한 출처 합쳐진 CSV 파일
* 점심과 저녁 식사 관련 다양한 컬럼 포함

---

2. 메뉴 컬럼 목록 지정

```python
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]
```

* 각 식사별 음식 이름을 담은 컬럼 리스트
* 각 날짜별 점심, 저녁 메뉴가 다수 포함됨

---

3. 메뉴 원-핫 인코딩

```python
all_menus = pd.concat([df[col] for col in menu_columns])
unique_menus = all_menus.dropna().unique()
menu_ohe = pd.DataFrame(0, index=df.index, columns=unique_menus)

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_ohe.at[idx, val] = 1
```

* 모든 메뉴 항목을 모아 고유 메뉴 리스트 생성
* 각 메뉴별로 데이터프레임 생성, 0/1로 메뉴 존재 여부 표시
* 범주형 메뉴 데이터를 이진형태로 변환해 머신러닝 모델이 처리하기 쉽게 함

---

4. 메뉴 빈도 인코딩

```python
menu_counter = Counter()
for col in menu_columns:
    menu_counter.update(df[col].dropna())

menu_word_encoded = pd.DataFrame(0, index=df.index, columns=unique_menus)
for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_word_encoded.at[idx, val] = menu_counter[val]
```

* 전체 데이터 내 각 메뉴 등장 빈도 집계
* 각 행의 메뉴에 대해 빈도수 값 할당
* 메뉴별 인기도 정보를 포함하여 모델이 메뉴별 영향도를 더 잘 학습할 수 있도록 함

---

5. 특성 생성 (Feature Engineering)

```python
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X = df[feature_cols].copy()
X['Season'] = X['Season'].astype('category').cat.codes
X['Day'] = X['Day'].astype('category').cat.codes
X = pd.concat([X, menu_ohe], axis=1)
X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)

y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']
```

* 휴일, 특별일, 온도 정보, 계절, 월, 요일 등 다양한 변수 선택
* 범주형 변수 숫자형으로 변환
* 메뉴 원-핫 인코딩 결과를 특


성에 합침

* 점심, 저녁 고객 수 컬럼을 별도 타겟 변수로 지정

---

6. 학습 및 평가

```python
# XGBoost 모델 정의 및 학습
xgb_lunch = XGBRegressor()
xgb_lunch.fit(X, y_lunch)

xgb_dinner = XGBRegressor()
xgb_dinner.fit(X, y_dinner)

# 학습 결과 저장 및 예측 결과 생성
pred_lunch = xgb_lunch.predict(X)
pred_dinner = xgb_dinner.predict(X)

df['pred_lunch'] = pred_lunch
df['pred_dinner'] = pred_dinner

df.to_csv(os.path.join(DATA_DIR, 'predictions_train_result.csv'), index=False)
```

* XGBoost 모델로 점심, 저녁 고객 수 각각 학습
* 훈련 데이터에 대해 예측 수행
* 결과를 CSV 파일로 저장

---

7. 모델 저장

```python
with open(os.path.join(MODEL_DIR, 'xgboost_lunch_model.pkl'), 'wb') as f:
    pickle.dump(xgb_lunch, f)

with open(os.path.join(MODEL_DIR, 'xgboost_dinner_model.pkl'), 'wb') as f:
    pickle.dump(xgb_dinner, f)
```

* 학습된 모델을 pkl 파일로 저장하여 추후 예측에 활용 가능

---
