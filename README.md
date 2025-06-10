
# SmartMealForecast 프로젝트 

## 1. 프로젝트 소개  
작품명: SmartMealForecast (식수 예측 시스템)  
최종 목표 : SmartMealForecast 프로젝트의 목표는 다음과 같은 요소들을 기반으로 하루 동안 준비해야 할 고객 수 또는 음식 수를 예측하는 시스템을 구축하는 것입니다:
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

## 5. 데이터 탐색 및 전처리
---
### - 데이터 탐색
* 총 1205행 데이터, 출처는 Dacon (데이터 고객/주문 : [Dacon 링크](https://dacon.io/competitions/official/235743/overview/description?utm_source=chatgpt.com))
* 날씨 데이터 출처 : [기상청 데이터](https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179)
* 휴일 및 특별일 데이터는 달력에서 수집
    ![image](https://github.com/user-attachments/assets/2163fde4-b0c9-45af-9b9d-cfce84f9cb9b)

* 엑셀을 이용해 분석하기 쉽게 데이터를 분리하고 필요한 원본 데이터를 병합하여 `merged_data.csv` 파일 생성
( Lunch_Menu, Dinner_Menu를 모델의 정확률 높이기 위해 'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'로 분리 )
  ![image](https://github.com/user-attachments/assets/7fb4064c-2c33-45d9-9e95-7351c3bd7053)


* 그래프(barplot_special_day_lunch_count_customers.png)를 보면 특별한 날에 고객 수가 확실히 많아지고, 월요일날들(2018-07-23, 2018-08-06 등)에도 고객 수가 늘어나는 것을 알 수 있다
 ![image](https://github.com/user-attachments/assets/e42c7a03-5800-410d-9830-d763d9db1c21)

* 날씨가 좋은 날에는 (15-20도) 점심 고객 수가 증가하는 경향이 있는 것으로 나타났다
 ![lineplot_2019_temp_vs_lunch](https://github.com/user-attachments/assets/34c76e26-0335-4ce3-b8c7-4ba1e49ab558)

### - 전처리

```python
# Data preprocessing / Tiền xử lý dữ liệu
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df.dropna(subset=['Date'], inplace=True)

[6]
# 4️ Check and handle missing values / Kiểm tra và xử lý giá trị thiếu
print("\nMissing Values Count:")
print(df.isnull().sum())

# Fill missing values with column mean / Điền giá trị thiếu bằng giá trị trung bình
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].mean(), inplace=True)
```

모델 선택: 
---
## 5. 코드


### 1) train.py 파일 코드 설명
---
* 주요 라이브러리 임포트
* `pandas`, `numpy` : 데이터 처리 및 수치 계산
* `joblib` : 모델 저장/불러오기
* `os` : 파일 경로 처리
* `Counter` : 메뉴 빈도 계산
* `sklearn` 관련 : 데이터 분할, 평가 지표, 랜덤포레스트 모델
* `xgboost` : XGBoost 모델

---


## 데이터 불러오기

```python
df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"))
```

* 합쳐진 데이터셋 csv 파일을 읽어와 `df`에 저장

---

## 메뉴 컬럼 리스트 정의

```python
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]
```

* 점심과 저녁 메뉴 항목들 컬럼명 리스트

---

## One-hot encoding 처리

```python
all_menus = pd.concat([df[col] for col in menu_columns])
unique_menus = all_menus.dropna().unique()
menu_ohe = pd.DataFrame(0, index=df.index, columns=unique_menus)

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_ohe.at[idx, val] = 1
```

* 메뉴의 모든 값들을 합쳐서 유니크 메뉴 아이템 추출
* 각 메뉴별로 one-hot 인코딩된 DataFrame 생성
* 메뉴가 있으면 해당 칼럼을 1로 표시

```python
merged_with_ohe = pd.concat([df, menu_ohe], axis=1)
merged_with_ohe.to_csv(os.path.join(DATA_DIR, "one_hot_encoded.csv"), index=False)
```

* 원본 데이터와 one-hot 인코딩된 메뉴 데이터를 합치고 저장

---

## 단어 빈도 인코딩 처리 (Word Frequency Encoding)

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

* 메뉴 아이템별 등장 빈도를 계산해 저장
* 각 데이터 행별 메뉴 아이템 등장 횟수를 값으로 할당

```python
merged_with_word_encoding = pd.concat([df, menu_word_encoded], axis=1)
merged_with_word_encoding.to_csv(os.path.join(DATA_DIR, "word_encoded.csv"), index=False)
```

* 원본 데이터와 단어 빈도 인코딩 데이터를 합쳐 저장

---

## Feature Engineering (특징 변수 가공)

```python
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X = df[feature_cols].copy()
X['Season'] = X['Season'].astype('category').cat.codes
X['Day'] = X['Day'].astype('category').cat.codes
X = pd.concat([X, menu_ohe], axis=1)
X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)
```

* 모델 입력 특징 변수 선택 및 복사
* `Season`, `Day` 범주형 변수 숫자 인코딩
* one-hot 메뉴 변수 합치기
* 컬럼명 중 특수문자 대체 (호환성 위해)

---

## 타깃 변수 설정

```python
y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']
```

* 점심, 저녁 각각의 식사 인원 수를 예측 목표로 설정

---

## 학습/테스트 데이터 분리

```python
X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)
```

* 데이터 80%는 학습용, 20%는 테스트용으로 분리

---

## 모델 학습

```python
model_lunch_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_dinner_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_lunch_rf = RandomForestRegressor(random_state=42)
model_dinner_rf = RandomForestRegressor(random_state=42)

model_lunch_xgb.fit(X_train, y_lunch_train)
model_dinner_xgb.fit(X_train, y_dinner_train)
model_lunch_rf.fit(X_train, y_lunch_train)
model_dinner_rf.fit(X_train, y_dinner_train)
```

* XGBoost와 랜덤포레스트 2종 모델 각각 점심과 저녁 예측용으로 학습

---

## 예측

```python
y_lunch_pred_xgb = model_lunch_xgb.predict(X_test)
y_dinner_pred_xgb = model_dinner_xgb.predict(X_test)
y_lunch_pred_rf = model_lunch_rf.predict(X_test)
y_dinner_pred_rf = model_dinner_rf.predict(X_test)
```

* 테스트 데이터에 대해 예측 수행

---

## 평가 함수 및 결과 계산

```python
def evaluate(true, pred):
    return mean_squared_error(true, pred), mean_absolute_error(true, pred)

mse_lunch_xgb, mae_lunch_xgb = evaluate(y_lunch_test, y_lunch_pred_xgb)
mse_dinner_xgb, mae_dinner_xgb = evaluate(y_dinner_test, y_dinner_pred_xgb)
mse_lunch_rf, mae_lunch_rf = evaluate(y_lunch_test, y_lunch_pred_rf)
mse_dinner_rf, mae_dinner_rf = evaluate(y_dinner_test, y_dinner_pred_rf)
```

* MSE, MAE 두 가지 지표로 모델 성능 평가

```python
baseline_lunch = np.full_like(y_lunch_test, y_lunch_train.mean())
baseline_dinner = np.full_like(y_dinner_test, y_dinner_train.mean())
mse_lunch_base, mae_lunch_base = evaluate(y_lunch_test, baseline_lunch)
mse_dinner_base, mae_dinner_base = evaluate(y_dinner_test, baseline_dinner)
```

* 평균값을 예측하는 기본선(baseline) 모델과 비교

---

## 평가 결과 저장

```python
with open(os.path.join(DATA_DIR, "evaluation_results.txt"), "w", encoding="utf-8") as f:
    f.write(" XGBoost Lunch\n")
    f.write(f"  MSE: {mse_lunch_xgb:.2f}, MAE: {mae_lunch_xgb:.2f}\n")
    f.write(" XGBoost Dinner\n")
    f.write(f"  MSE: {mse_dinner_xgb:.2f}, MAE: {mae_dinner_xgb:.2f}\n\n")
    f.write(" Random Forest Lunch\n")
    f.write(f"  MSE: {mse_lunch_rf:.2f}, MAE: {mae_lunch_rf:.2f}\n")
    f.write(" Random Forest Dinner\n")
    f.write(f"  MSE: {mse_dinner_rf:.2f}, MAE: {mae_dinner_rf:.2f}\n\n")
    f.write(" Baseline Lunch\n")
    f.write(f"  MSE: {mse_lunch_base:.2f}, MAE: {mae_lunch_base:.2f}\n")
    f.write(" Baseline Dinner\n")
    f.write(f"  MSE: {mse_dinner_base:.2f}, MAE: {mae_dinner_base:.2f}\n")
```

* 텍스트 파일로 평가 결과 기록

---

## 예측 결과를 원본 데이터프레임에 추가하고 저장

```python
df_preds = df.copy()
df_preds['lunch_pred_xgb'] = np.nan
df_preds['dinner_pred_xgb'] = np.nan
df_preds['lunch_pred_rf'] = np.nan
df_preds['dinner_pred_rf'] = np.nan
df_preds['baseline_lunch_pred'] = np.nan
df_preds['baseline_dinner_pred'] = np.nan

df_preds.loc[X_test.index, 'lunch_pred_xgb'] = y_lunch_pred_xgb
df_preds.loc[X_test.index, 'dinner_pred_xgb'] = y_dinner_pred_xgb
df_preds.loc[X_test.index, 'lunch_pred_rf'] = y_lunch_pred_rf
df_preds.loc[X_test.index, 'dinner_pred_rf'] = y_dinner_pred_rf
df_preds.loc[X_test.index, 'baseline_lunch_pred'] = baseline_lunch
df_preds.loc[X_test.index, 'baseline_dinner_pred'] = baseline_dinner

df_preds.to_csv(os.path.join(DATA_DIR, "predictions_train_result.csv"), index=False)
```

* 예측값을 NaN으로 초기화 후 테스트 인덱스 위치에 예측 결과를 넣음
* 저장하여 결과 비교 가능

---

## 가공된 데이터 및 타깃 변수 저장

```python
X.to_csv(os.path.join(DATA_DIR, "x_encoded.csv"), index=False)
pd.DataFrame(y_lunch).to_csv(os.path.join(DATA_DIR, "y_lunch.csv"), index=False)
pd.DataFrame(y_dinner).to_csv(os.path.join(DATA_DIR, "y_dinner.csv"), index=False)
```

* 입력 특징과 타깃 데이터를 각각 CSV 파일로 저장

---

## 학습한 모델 저장

```python
joblib.dump(model_lunch_xgb, os.path.join(MODEL_DIR, "xgboost_lunch_model.pkl"))
joblib.dump(model_dinner_xgb, os.path.join(MODEL_DIR, "xgboost_dinner_model.pkl"))
joblib.dump(model_lunch_rf, os.path.join(MODEL_DIR, "ranfor_lunch_model.pkl"))
joblib.dump(model_dinner_rf, os.path.join(MODEL_DIR, "ranfor_dinner_model.pkl"))
```

* 학습 완료한 XGBoost, Random Forest 모델을 각각 저장

---
## 2) prediction.py 파일 코드 설명
---
## 라이브러리	

* pandas, numpy, joblib, os, collections 모듈 임포트
* pandas: 데이터 처리
* numpy: 수치 계산
* joblib: 저장된 모델 로딩
* os: 파일 경로 처리
* Counter: 데이터 빈도 계산용


```python
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]
```

* 점심과 저녁 메뉴 관련 컬럼 리스트 정의

---

```python
all_menus = pd.concat([df[col] for col in menu_columns])
unique_menus = all_menus.dropna().unique()
menu_ohe = pd.DataFrame(0, index=df.index, columns=unique_menus)

for col in menu_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            menu_ohe.at[idx, val] = 1
```

* 메뉴 컬럼들의 값을 모두 합쳐 고유 메뉴 리스트 생성
* 각 메뉴를 열로 하는 원-핫 인코딩 데이터프레임 생성
* 각 행, 각 메뉴에 대해 해당 메뉴가 있으면 1로 표시

---

```python
feature_cols = ['Holiday', 'special_day', 'Avg_Temp', 'Max_Temp', 'Min_Temp',
                'Temp_Range', 'Season', 'Month', 'Day']

X_all = df[feature_cols].copy()
X_all['Season'] = X_all['Season'].astype('category').cat.codes
X_all['Day'] = X_all['Day'].astype('category').cat.codes
X_all = pd.concat([X_all, menu_ohe], axis=1)
X_all.columns = X_all.columns.str.replace(r'[\[\]<>]', '_', regex=True)
```

* 모델 입력 특성 리스트 지정
* 'Season', 'Day'를 범주형으로 변환 후 코드화
* 원-핫 인코딩 메뉴 컬럼과 병합
* 컬럼명 내 특수문자 대체

---

```python
model_lunch_xgb = joblib.load(os.path.join(ROOT_DIR, "models", "xgboost_lunch_model.pkl"))
model_dinner_xgb = joblib.load(os.path.join(ROOT_DIR, "models", "xgboost_dinner_model.pkl"))
model_lunch_rf = joblib.load(os.path.join(ROOT_DIR, "models", "ranfor_lunch_model.pkl"))
model_dinner_rf = joblib.load(os.path.join(ROOT_DIR, "models", "ranfor_dinner_model.pkl"))
```

* 저장된 머신러닝 모델 4개 로드 (XGBoost, 랜덤포레스트 각각 점심/저녁용)

---

```python
df['Lunch_Pred_XGB'] = model_lunch_xgb.predict(X_all).round().astype(int)
df['Dinner_Pred_XGB'] = model_dinner_xgb.predict(X_all).round().astype(int)
df['Lunch_Pred_RF'] = model_lunch_rf.predict(X_all).round().astype(int)
df['Dinner_Pred_RF'] = model_dinner_rf.predict(X_all).round().astype(int)
```

* 각 모델별 점심, 저녁 인원수 예측
* 예측값 반올림 후 정수 변환

---

```python
lunch_mean = int(round(df['Lunch_Count'].mean()))
dinner_mean = int(round(df['Dinner_Count'].mean()))
df['Lunch_Pred_Baseline'] = lunch_mean
df['Dinner_Pred_Baseline'] = dinner_mean
```

* 점심, 저녁 인원수 평균값 계산 후 베이스라인 예측값으로 설정

---

```python
save_path = os.path.join(RESULTS_DIR, "predictions_all_data.csv")
df.to_csv(save_path, index=False)

print("Saved to :", save_path)
print(df[['Lunch_Count', 'Lunch_Pred_XGB', 'Lunch_Pred_RF', 'Lunch_Pred_Baseline']].head())
```

* 결과 CSV 파일로 저장
* 저장 위치 및 예측값 일부 출력

---

## 요약

* 데이터 로딩 → 메뉴 원-핫 인코딩 → 특성 데이터 준비 → 모델 로딩 → 예측 수행 → 결과 저장
* XGBoost와 랜덤포레스트 모델로 점심/저녁 인원수 예측
* 단순 평균 기반 베이스라인과 비교 평가 가능

```
## 6. 실행 방법  
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
## 7. 실행결과 분석
---
- 예측 결과: 
![image](https://github.com/user-attachments/assets/fb8d1c12-a921-4036-9d2d-8ba202f6c9b8)

- 에러 지표:
![mae_comparison_improved_chart](https://github.com/user-attachments/assets/37a851ea-7a81-4298-ae62-d568e774e94a)

- 결론
XGB 모델이 점심/저녁 모두에서 가장 낮은 MAE → 가장 정확한 예측.
RF도 성능이 괜찮지만 XGB보다 약간 낮음.
Baseline은 오차가 크고, 정규화 MAE도 높음 → 단순 평균 예측은 효과적이지 않음  
결론: XGB 모델이 현재 데이터에 대해 가장 신뢰할 수 있는 예측 모델입니다.

---

## 8. Author: Nguyen Thi Dung ( 응웬티둥)
---
