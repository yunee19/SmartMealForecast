### **[03/21 팀미팅 아젠다]**  

#### **1. 작품 간략히 소개**  
- **SmartMealForecast**: 머신러닝을 활용하여 구내식당/레스토랑의 일일 고객 수 또는 음식 수요를 예측하는 시스템  
- **주요 데이터**: 고객/주문 데이터, 요일, 날씨(기온, 강수량 등), 공휴일  

#### **2. 현재 진행하는 부분**  
- **데이터 수집**  
  - **고객/주문 데이터:**  
    - **출처:**  
      - Kaggle
          - **File: Dataset_02** : [Food Choices Dataset](https://www.kaggle.com/datasets/borapajo/food-choices?utm_source=chatgpt.com)
          - **File: Dataset_03** : [Student Food Survey Dataset](https://www.kaggle.com/datasets/mlomuscio/student-food-survey?utm_source=chatgpt.com)
      - Dacon  
        - **Dataset_01:** Dacon Competition 관련 데이터  [Dacon Competition](https://dacon.io/competitions/official/235743/overview/description)
    - **진행 상황:** 3개의 데이터셋을 찾았으나, Dataset_01가 가장 적절할 것 판단.

  - **날씨 데이터:**  
    - **출처:**  기상청 지상자료개방포털  [KMA Climate](https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179) 
    - **문제:** 데이터를 표 형식으로 처리하는 과정에서 오류 및 불완전한 데이터 발생  
    - **해결 방안:** 한글 지원 소프트웨어 설치 및 변환 도구 활용 예정  
  - **공휴일 데이터:**  
    - **출처:** 네이버 (공휴일 정보 확인)  
    - **문제:** API 사용 시 로그인 필요, 자동화 어려움  
    - **해결 방안:** 초기에는 수동으로 데이터를 CSV로 저장, 추후 API 활용 검토  

#### **3. 현재 문제/고민하는 부분**  
✅ **데이터 선정 문제**    
- 기존 데이터셋이 현재 프로젝트의 목표(구내식당/학생식당 수요 예측)에 적합한지 검토 필요.  
- **추가 데이터가 필요할까?**   

✅ **예측 모델 추천**  
- **어떤 모델이 가장 적절할까?**  
  - 기본적인 (**선형 회귀(Linear Regression)**, **랜덤 포레스트(Random Forest)**, **XGBoost**, **LSTM**(chatgpt 권장해준)) 중 어떤 것을 고려해야 할까?  
  - 비슷한 연구 사례에서는 어떤 모델을 사용했는지 조사 필요.  (castboostRgressor,cohere embedding..)
  
  - ###  Chat GPT 추천 **스마트밀포캐스트(SmartMealForecast) 예측 모델 비교**  

| 모델 | 장점 | 단점 | 적절한 경우 |
|------------------------|---------|------------|--------------------|
| **RandomForestRegressor** | - 사용하기 쉽고 하이퍼파라미터 튜닝이 거의 필요 없음.  - 여러 개의 결정 트리를 사용하여 과적합(Overfitting) 방지. | - 데이터가 많아질수록 느려짐.  - 시계열 데이터 처리에 적합하지 않음. | - **구내식당/레스토랑의 주문 데이터를 정형 데이터(tabular data) 형태로 사용할 때** 적합. |
| **XGBoost** | - 빠른 연산 속도, 메모리 최적화. - RandomForest보다 높은 예측 성능을 보임. | - 하이퍼파라미터 조정이 필요함. - 이상치(Outlier)에 민감함. | - **정확도가 중요한 경우, 대용량 데이터 활용 시 추천**. |
| **LSTM (Long Short-Term Memory)** | - 시계열 데이터를 다룰 수 있음.  - 시간에 따른 트렌드 및 주기적 패턴을 학습 가능. | - 많은 데이터가 필요함. - 학습 속도가 느리고 GPU 필요. | - **과거 데이터(날짜, 요일, 계절 등)를 기반으로 음식 수요를 예측할 때 적절함**. |
| **CatBoostRegressor** | - 범주형 데이터(Categorical Data) 처리가 강력함. - 원-핫 인코딩(One-hot encoding) 없이도 작동 가능. - 작은 데이터셋에서도 성능이 좋음. | - 대용량 데이터에서는 XGBoost보다 느릴 수 있음. - 하이퍼파라미터 튜닝이 필요함. | - **날씨, 공휴일, 요일 등의 범주형 데이터를 포함한 식사 예측 모델에 적합**. |

---

### **스마트밀포캐스트(SmartMealForecast) 프로젝트에 적합한 모델은?**
✔ **정형 데이터(tabular data)**(날씨, 요일, 공휴일 등)를 활용하는 경우 → **CatBoostRegressor 또는 XGBoost** 추천  
✔ **주별, 월별 패턴을 분석해야 하는 경우** → **LSTM** 추천  
✔ **빠르고 간단한 모델이 필요할 경우** → **RandomForestRegressor** 추천  

**어떤 모델을 먼저 적용해볼까요?** 🚀


✅ **음식 이름을 벡터화할지 여부**  
- 인터넷에서 찾은 프로젝트에서는 **음식 이름을 벡터화**하여 예측을 진행했음.  
- 우리 프로젝트에서도 음식 이름을 벡터화하는 것이 필요한가?  
- 아니면 음식 종류별 카테고리를 구분하는 것이 더 나을까?  

#### **4. 향후 계획**  
- **4주차(03/28)까지** 데이터 전처리 및 정제 완료  
- 예측 모델 조사 및 비교 분석 → 가장 적합한 모델 선정  
- 필요시 추가 데이터 수집 검토  
