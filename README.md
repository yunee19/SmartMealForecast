
# SmartMealForecast (식수 예측)  

## 1. 프로젝트 소개  

**최종 목표** 

본 프로젝트의 목표는 과거 고객/식수 데이터, 날씨, 특별일 정보, 사내 인원 정보 등을 기반으로 하루 동안 준비해야 할 점심·저녁 식수량을 예측하는 것입니다. 이를 통해 음식 낭비를 줄이고 부족·과잉 공급을 방지하여 운영 효율성을 높이고 고객 만족도를 향상시키고자 합니다.

**식수량 예측의 중요성**

- 정확한 수요 예측 → 음식 낭비 감소

- 인력·재료 비용 최적화

- 고객 만족도 향상

**사용 방법론**  

- `XGBoost`, `Random Forest` 등 비선형적·복잡한 패턴을 잘 처리할 수 있는 머신러닝 알고리즘 사용
- 메뉴 정보 인코딩(원-핫, 빈도 기반) 및 직원 수 예측 등 특화된 Feature Engineering 적용
  


## 2. 데이터 및 구조  

### 2.1 데이터 수집

* 총 1205행 데이터, 출처는 Dacon (데이터 고객/주문 : [Dacon 링크](https://dacon.io/competitions/official/235743/overview/description?utm_source=chatgpt.com))
* 날씨 데이터 출처 : [기상청 데이터](https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179)
* 휴일 및 특별일 데이터는 달력에서 수집
* 칼로리 정보 : ChatGPT로 수집
  

### 2.2 프로젝트 폴더 구조
```

SmartMealForecast/
├── data/                        ← 원본 데이터 및 전처리 데이터
│   ├── merged\_data\_2\_kcal.csv   ← 데이터  
│   └── ... 등등


├── predictions/                 ← 예측 수행 관련 스크립트
│   ├── train\_2.py                ← 모델 훈련
│   ├── prediction\_2.py           ← 전체 데이터에 대한 예측 실행
│   └── predict\_by\_date.py             ← 지정 날짜에 예측
│   └── ... 등등

├── models/                     ← 학습된 모델 저장
│   ├── xgboost\_dinner\_model\_2.pkl
│   └── xgboost\_lunch\_model\_2.pkl
|   └──... 등등

├── evaluation/                 ← 평가 결과 및 시각화 자료
│   ├── evaluation\_2.py           ← MAE, Normalized MAE 계산 및 그래프 생성 스크립트
│   ├── graph\_2.py                ← 다양한 시각화 그래프 생성 스크립트
│   └──... 등등

├── prediction\_result/          ← 예측 결과 저장
│   └── predictions\_all\_data\_2.csv
│   └── train\_2\_predictions.csv
│   └──... 등등

└── README.md                   ← 프로젝트 설명서 (현재 파일)

````


## 3. 데이터 분석

### 3.1 데이터 기본 현황

**데이터셋 : merged_data_2_kcal.csv**

**총 1,205행 데이터**

| 그룹                     | 예시 열                                                                                                   | 의미                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------- |
| **날짜 & 계절 정보**     | `Date`, `Week_Day`, `Year`, `Month`, `Day`, `Season`                                                      | 날짜, 요일, 계절                  |
| **인사(인원) 정보**      | `Total_Emp`, `Actual_Emp`, `Leave_Emp`, `Trip_Emp`, `OT_Approved`, `Remote_Emp`                           | 총 인원, 휴가, 출장, OT, 재택 등  |
| **점심 메뉴**            | `Lunch_Rice`, `Lunch_Soup`, `Lunch_Main_Dish`, `Lunch_Side_Dish_1`, `Lunch_Kimchi` … + kcal 해당           | 각 메뉴명 및 칼로리               |
| **저녁 메뉴**            | `Dinner_Rice`, `Dinner_Soup`, `Dinner_Main_Dish`, `Dinner_Side_Dish_1`… + kcal 해당                      | 각 메뉴명 및 칼로리               |
| **기타**                 | `Station` (위치: 서울), `Avg_Temp`, `Max_Temp`, `Min_Temp`, `Temp_Range`, `Special_Day`                  | 날씨 및 특수일 정보               |

### 3.2 주요 인사이트

* **특별일에 고객 수 급증**

<img width="1600" height="600" alt="barplot_special_day_lunch_count_customers" src="https://github.com/user-attachments/assets/3c267b3b-e875-48a8-9f77-b618cbaec283" />


* **요일 특정 : 월요일에 고갯수 급증**

<img width="1000" height="600" alt="avg_customers_by_weekday" src="https://github.com/user-attachments/assets/457c37f7-5a0f-43dc-930e-937398b35a77" />


* **쾌적한 날씨에 점심 고객 수가 증가**

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/b4caa481-23b6-4041-adc6-3e26d1195576" />

* **계절 따라 고객증가**

<img width="800" height="600" alt="barplot_season_total_customers" src="https://github.com/user-attachments/assets/f8c41a3e-897b-46eb-ac91-1c6b85d3ee80" />


## 4. 전처리 및 특징 엔지니어링

1. **결측치 처리 (Missing Values)**  
   - 일부 날짜·메뉴 데이터에 결측치가 존재하여 평균값 또는 최빈값으로 대체하거나, 예측에 필요 없는 경우 해당 행(row)을 제거하였다.  
   - 예: 특정 날 메뉴 칼로리 값이 비어 있는 경우 동일 메뉴의 평균 칼로리를 사용.

2. **이상치 처리 (Outlier Handling)**  
   - 극단적으로 높거나 낮은 고객 수·메뉴 주문 수는 모델을 왜곡할 수 있으므로 IQR(사분위 범위) 및 Z-score를 활용해 이상치를 탐지·조정하였다.

3. **범주형 변수 인코딩 (Categorical Encoding)**  
   - 메뉴명, 요일, 날씨 등 범주형(categorical) 데이터를 원-핫 인코딩(One-Hot Encoding) 또는 레이블 인코딩(Label Encoding)하여 모델이 이해할 수 있도록 수치형 데이터로 변환하였다.

4. **스케일링 (Scaling)**  
   - 변수 간 스케일 차이를 줄이기 위해 표준화(Standardization) 또는 정규화(Normalization)를 적용해 모델 학습의 안정성을 확보하였다.

5. **파생 변수 생성 (Feature Creation)**  
   - 고객 수·메뉴 패턴을 더 잘 설명하기 위해 기존 변수에서 새로운 특징을 생성하였다.
     - 요일별 평균 고객 수, 계절(봄/여름/가을/겨울) 변수 추가  
     - 공휴일 여부(holiday flag) 변수 추가  
     - 날씨(강수량·기온)를 결합한 ‘체감 기온’ 변수 생성

6. **시간 시퀀스 특성 반영 (Time-series Features)**  
   - 이전 일·주·월의 고객 수 이동평균(lag features, rolling mean)과 증감률을 계산하여 수요 예측의 패턴성을 높였다.



## 5. 방법론 및 구현


### 5.1 train_2.py 파일 코드 설명

* 주요 라이브러리 임포트
* `pandas`, `numpy` : 데이터 처리 및 수치 계산
* `joblib` : 모델 저장/불러오기
* `os` : 파일 경로 처리
* `Counter` : 메뉴 빈도 계산
* `sklearn` 관련 : 데이터 분할, 평가 지표, 랜덤포레스트 모델
* `xgboost` : XGBoost 모델
* `gensim.models` : Word2Vec


**(1) 메뉴 컬럼 리스트 정의**
```python
menu_columns = [
    'Lunch_Rice', 'Lunch_Soup', 'Lunch_Main_Dish', 'Lunch_Side_Dish_1',
    'Lunch_Side_Dish_2', 'Lunch_Drink', 'Lunch_Kimchi', 'Lunch_side_Dish_3',
    'Dinner_Rice', 'Dinner_Soup', 'Dinner_Main_Dish', 'Dinner_Side_Dish_1',
    'Dinner_Side_Dish_2', 'Dinner_Side_Dish_3', 'Dinner_Drink', 'Dinner_Kimchi'
]
```
**(2) 데이터 로드 및 메뉴 컬럼 정의**
- 메뉴 관련 컬럼들을 추후 Word2Vec 입력 및 임베딩 대상(각 날짜의 메뉴 목록)으로 사용.
  
**(3) 기본 특성 선택(Feature cols) 및 Season 인코딩**
- 모델 입력으로 사용할 기본 특성을 선택
- Season을 범주형 코드로 변환하여 수치형 모델에서 사용 가능하도록 함

**(4) `Pre_Special_Day` (특별일 전/후 변수)**
- 특별일 전(전날) 효과를 반영

**(5) `Actual_Emp`(실제 근무자) 예측 모델 (내부 피처로 사용)**
  - Actual_Emp(실제 출근자 수)가 누락되거나 실시간 수집이 불가능한 경우를 대비해, 다른 특성으로 Actual_Emp를 예측하고 이를 후속 feature로 사용함

**(6) `Emp_Ratio` 생성 (예측된 실제 인원 / 총 인원)**
- 근무율(예: 실제 출근자 비율)을 특성으로 포함하여 고객 수와의 상관관계를 반영

**(7) 시간(주기성) 인코딩: Sin/Cos**
- Month, Day의 순환적(주기적) 속성 반영 — 머신러닝 모델이 계절성/주기성을 더 잘 잡게 함.

**(8) 요일(WeekDay) sin/cos 인코딩**
- 요일(월~금)에 대한 주기성 반영

**(9) 메뉴 Word2Vec 임베딩**

```python
menu_sentences = []
for idx, row in df.iterrows():
    sentence = [str(row[col]) for col in menu_columns if pd.notna(row[col])]
    if sentence:
        menu_sentences.append(sentence)

w2v_model = Word2Vec(sentences=menu_sentences, vector_size=100, window=5, min_count=1, workers=4, seed=42)
w2v_model.save(os.path.join(MODEL_DIR, "w2v_menu.model"))

```
- 메뉴 텍스트(예: 반찬 이름 등)를 분산 임베딩(Word2Vec)으로 표현하여 의미적 유사도를 특성으로 반영.
- `vector_size=100` 은 각 메뉴를 100차원 벡터로 표현.
- `min_count=1`로 설정되어 있어 등장 한 번만 해도 임베딩에 포함됨(희소한 메뉴까지 반영됨).

  
**(10) 특성 결합 및 컬럼명 정리**
  - 기본 특성 + 메뉴 임베딩을 합쳐 최종 입력 매트릭스 X를 구성
 
**(11) 타깃 지정 및 학습/테스트 분할**

```python
y_lunch = df['Lunch_Count']
y_dinner = df['Dinner_Count']

X_train, X_test, y_lunch_train, y_lunch_test = train_test_split(X, y_lunch, test_size=0.2, random_state=42)
_, _, y_dinner_train, y_dinner_test = train_test_split(X, y_dinner, test_size=0.2, random_state=42)
```
**(12) XGBoost 학습 함수 및 모델 학습**

```python
def train_xgb(X_tr, y_tr, X_val, y_val):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
              'seed': 42, 'max_depth': 8, 'eta': 0.1}

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=evals, early_stopping_rounds=50, verbose_eval=50)
    return model

model_lunch_xgb = train_xgb(X_train, y_lunch_train, X_test, y_lunch_test)
model_dinner_xgb = train_xgb(X_train, y_dinner_train, X_test, y_dinner_test)

```
- `X_tr`, `y_tr` : 학습(Train)용 입력 데이터와 정답(라벨), `X_val`, `y_val` : 검증(Validation)용 입력 데이터와 정답(라벨)( 학습 데이터와 검증 데이터를 받아서 XGBoost 회귀 모델을 만들어 반환합니다)
  
- `DMatrix`로 변환 
      - XGBoost는 자체 데이터 구조인 DMatrix를 사용합니다.
      - DMatrix는 메모리와 계산 속도를 최적화한 자료구조로, 모델 학습·예측 때 더 빠르게 동작하게 해줍니다.
      - `label=y_tr`로 실제 정답 값을 함께 지정합니다.
  
- 하이퍼파라미터 설정
      - `objective` : 학습 목적 함수. 'reg:squarederror'는 회귀(연속값 예측)용 제곱오차를 의미합니다.
      - `eval_metric` : 평가 지표. rmse는 root mean squared error(평균제곱근오차)입니다.
      - `seed` : 랜덤 시드 고정. 재현성(같은 결과)을 위해 사용합니다.
      - `max_depth` : 트리 최대 깊이. 클수록 더 복잡한 모델이 됩니다(과적합 주의).
      - eta : 학습률(learning rate). 한 번의 학습 스텝에서 파라미터가 얼마나 업데이트될지 결정.
  
- 학습과 검증 데이터 지정
    - 학습 과정에서 학습(train) 데이터와 검증(eval) 데이터의 오차를 함께 확인하기 위해 튜플로 지정.
    - `xgb.train()`이 진행될 때 각 단계마다 두 데이터셋의 RMSE를 출력합니다.
 
- 모델 학습
    - `params` : 앞에서 지정한 하이퍼파라미터.
    - `num_boost_round=500` : 최대 500회(부스팅 라운드)까지 반복 학습.
    - `evals=evals` : 매 라운드마다 학습·검증 오차 출력.
    - `early_stopping_rounds=50` : 검증 오차가 50번 연속 개선되지 않으면 학습 조기 종료(과적합 방지).
    - `verbose_eval=50` : 50라운드마다 학습 상황을 출력.
    - 이렇게 하면 XGBoost 모델이 학습되고, 조기 종료가 적용되어 불필요한 반복을 줄일 수 있다.
 
- 실제 사용
    - `model_lunch_xgb` : 점심 고객 수(Lunch_Count) 예측 모델
    - `model_dinner_xgb` : 저녁 고객 수(Dinner_Count) 예측 모델

**(13) 예측 및 평가**

```python
y_lunch_pred = model_lunch_xgb.predict(xgb.DMatrix(X_test))
y_dinner_pred = model_dinner_xgb.predict(xgb.DMatrix(X_test))

def evaluate(true, pred):
    return mean_squared_error(true, pred), mean_absolute_error(true, pred)

mse_lunch, mae_lunch = evaluate(y_lunch_test, y_lunch_pred)
mse_dinner, mae_dinner = evaluate(y_dinner_test, y_dinner_pred)

```
- 테스트셋에서 예측 후 MSE, MAE 계산

**(14) Baseline(평균 예측) 생성**

Baseline은 과거 점심과 저녁 식사 고객 수를 평균값으로 예측합니다. 이 모델은 더 복잡한 모델들과 비교하기 위한 기준점으로 만들어졌다

**(15) 평가/예측/모델 저장**
```python
eval_df.to_csv(os.path.join(EVALUATION_DIR, "train_2_evaluation.csv"), index=False)
pred_df.to_csv(os.path.join(PRED_DIR, "train_2_predictions.csv"), index=False)

joblib.dump(model_lunch_xgb, os.path.join(MODEL_DIR, "xgboost_lunch_model_2.pkl"))
joblib.dump(model_dinner_xgb, os.path.join(MODEL_DIR, "xgboost_dinner_model_2.pkl"))
```
- XGBoost 모델 저장

### 5.2 prediction_2.py 파일 코드 설명( 모든 데이터 가지고 실행)

**라이브러리**	

* `pandas`, `numpy`: 표(데이터프레임)와 수치 연산
* `joblib`: 학습된 모델 불러오기/저장
* `os`: 파일 경로 처리
* `xgboost`: 예측 모델
* `Word2Vec`: 메뉴 이름을 벡터로 변환(임베딩)

**모델 로드 (점심/저녁)**
```python
model_lunch_xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_lunch_model_2.pkl"))
model_dinner_xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_dinner_model_2.pkl"))
```

**예측**
```python
df['Pred_Lunch_XGB'] = model_lunch_xgb.predict(xgb.DMatrix(X)).round().astype(int)
df['Pred_Dinner_XGB'] = model_dinner_xgb.predict(xgb.DMatrix(X)).round().astype(int)
```

**결과 저장**
```python
save_path = os.path.join(RESULTS_DIR, "predictions_all_data_2.csv")
df.to_csv(save_path, index=False)
```
**실행결과 예**

식수 예측의 결과: 

| Date       | 실제점심 | 실제저녁 | 점심_XGB | 저녁_XGB |
|------------|------------|--------------|----------------|-----------------|
| 2021-01-13 | 913        | 360          | 913            | 360             |
| 2021-01-14 | 1038       | 439          | 822            | 405             |
| 2021-01-15 | 702        | 277          | 638            | 324             |
| 2021-01-18 | 1277       | 460          | 1254           | 477             | 
| 2021-01-19 | 1038       | 414          | 1038           | 414             | 



## 6. 실행 방법  및 실행결과 분석
### 6.1 실행 방법

1. 필요한 데이터 준비: `data/` 폴더에 merged_data_2_kcal.csv 데이터 저장
2. 모델 훈련 및 예측 실행
3. 사용자 지정 날짜로 실행할 경우 predict_by_date.py 사용

```bash
python prediction/train_2.py  
python prediction/prediction_2.py
python prediction/predict_by_date.py
```
### 6.2 실행결과 분석

* **실행결과 예**

| Date       | 실제점심 | 실제저녁 | 점심_XGB | 저녁_XGB |
|------------|------------|--------------|----------------|-----------------|
| 2021-01-13 | 913        | 360          | 913            | 360             |
| 2021-01-14 | 1038       | 439          | 822            | 405             |
| 2021-01-15 | 702        | 277          | 638            | 324             |
| 2021-01-18 | 1277       | 460          | 1254           | 477             | 
| 2021-01-19 | 1038       | 414          | 1038           | 414             | 


* **predict_by_date.py 살행결과**


<img width="969" height="753" alt="image" src="https://github.com/user-attachments/assets/9e76db2c-386c-423e-8871-36aa52f5ceed" />




* **prediction_2.py 살행결과**

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/e46587ef-ab63-4f3d-8c74-70fd63847009" />

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/a2711054-e945-45fe-91ec-0d55ddae5624" />



* **에러 지표**

  
<img width="1600" height="700" alt="mae_lunch_dinner_separate" src="https://github.com/user-attachments/assets/8353d10e-303b-434d-ab76-88052ef1b8a3" />

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/c350cb7c-d4ab-4c80-bd57-6ba419aac345" />

**비교 해보기**

* Baseline은 과거 점심과 저녁 식사 고객 수를 평균값으로 예측합니다. 이 모델은 더 복잡한 모델들과 비교하기 위한 기준점으로 만들어졌으며, 선택된 모델이 실제로 개선 효과가 있는지 평가하는 데 사용됩니다.

* 1학기 동안 Random Forest (RF) 모델을 사용했습니다. RF는 입력 변수와 출력 간의 비선형 관계를 학습할 수 있습니다. RF는 **MAE = 5.89%**로 꽤 정확한 결과를 보여주었으며, baseline보다 크게 개선되었습니다.

* XGBoost는 강력한 부스팅 알고리즘으로 구현되었으며, **점심과 저녁 모두에서 가장 낮은 MAE = 2.52%**를 기록했습니다. 이는 RF 대비 거의 절반 수준으로 MAE가 감소한 것이며, XGB가 데이터의 특성을 더 효율적으로 활용하여 예측 정확도를 크게 향상시킴을 보여줍니다. 실제로 XGBoost는 낮은 MAE와 정규화 MAE 수치로 보여줍니다

* Baseline (과거 평균) → MAE 20.23% (정확도 낮음)

* RF → MAE 5.89% (개선됨, 하지만 XGB보다 낮음)

* XGBoost (핵심 모델) → MAE 2.52% (가장 정확, RF 대비 약 1/2 수준으로 감소)

* XGBoost는 점심과 저녁 시간대 모두에서 뛰어난 성능을 보여주며, RF보다 예측력이 더 우수합니다. 프로젝트에서 핵심 모델로 선정된 이유가 여기에 있습니다.

* 이처럼 SmartMealForecast 프로젝트는 날씨, 휴일, 메뉴 종류 등의 다양한 현실적 요인을 바탕으로
일일 식수량을 예측하는 시스템을 성공적으로 구축하였습니다.

* GBoost와 Random Forest와 같은 최신 머신러닝 모델을 활용하여 비선형적이며 다차원적인 데이터를 효과적으로 처리할 수 있었으며, One-hot Encoding, 빈도 기반 인코딩 등 다양한 특성 인코딩 기법을 적용하고 MAE, MSE 등 지표를 기반으로 평가를 수행한 결과, 현장에 적용 가능한 신뢰도 높은 결과를 도출할 수 있었습니다.
  
## 결론

**SmartMealForecast** 프로젝트는 날씨, 요일, 계절, 메뉴 등 실제 요소를 기반으로 점심과 저녁 고객 수를 예측하는 모델을 성공적으로 구축하였습니다. 학습과 평가 과정을 통해 다음과 같은 결과를 얻었습니다:

1. **XGBoost 모델 성능**
   - XGBoost를 이용한 점심 및 저녁 예측은 **MAE와 MSE가 baseline(평균)보다 현저히 낮아**, 모델이 데이터의 중요한 특성을 잘 학습했음을 보여줍니다.  
   - 예를 들어, 특별한 날과 기온, 요일(weekday)이 고객 수에 직접적인 영향을 미치며, XGBoost는 이러한 경향을 정확하게 반영합니다.

2. **입력 변수의 영향** 
   - `Special_Day`, `Avg_Temp`, `Total_Emp`와 같은 변수와 **메뉴 임베딩**, 그리고 **요일(Weekday) 인코딩**을 결합하여 모델이 메뉴·날짜·요일과 고객 수 사이의 복잡한 관계를 잘 파악할 수 있습니다.  
   - 또한 `Emp_Ratio` 및 주/월에 대한 sin/cos 인코딩을 적용하여 계절 및 요일별 패턴을 더욱 정교하게 반영하였습니다.

3. **실제 적용 가능성** 
   - 정확한 고객 수 예측은 주방과 관리자가 **재료, 인력, 메뉴 계획**을 더 효율적으로 수행할 수 있도록 돕습니다.
   - 시스템은 일일 고객 수 예측과 메뉴 추천, 칼로리 계산까지 제공하며, 실제 급식 관리나 구내식당 운영에 즉시 적용 가능한 실용적인 도구입니다.

XGBoost 모델과 피처 엔지니어링, 메뉴 임베딩, 요일 인코딩의 결합은 일일 식수 수요 예측에서 뛰어난 성능을 보여주었으며, baseline보다 우수합니다. 이 프로젝트는 실제 급식 관리나 구내식당 운영에 바로 적용 가능한 실용적인 도구를 제공합니다.



## 7. Author: Nguyen Thi Dung ( 응웬티둥)

