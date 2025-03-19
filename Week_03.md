# **진행 상황 보고서 (3주차, 03/21)**

## 1. **데이터 수집**
- **고객/주문 데이터:**
  - **데이터 출처:**
    - **Kaggle:**
      - **File: Dataset_02** : [Food Choices Dataset](https://www.kaggle.com/datasets/borapajo/food-choices?utm_source=chatgpt.com)
      -  **File: Dataset_03** : [Student Food Survey Dataset](https://www.kaggle.com/datasets/mlomuscio/student-food-survey?utm_source=chatgpt.com)

    - **Dacon:** [Dacon Competition](https://dacon.io/competitions/official/235743/overview/description)에서 유용한 데이터셋을 찾았으며, 이를 본 프로젝트에 적용할 수 있습니다.
  
  - **해결 방안:** 위의 출처에서 데이터를 다운로드하여 분석을 위한 전처리를 진행할 예정입니다.

- **날씨 데이터:** 파일명: **weather_data_korea_2024.csv**
  - **데이터 출처:** 날씨 데이터는 [KMA Climate](https://data.kma.go.kr/climate/RankState/selectRankStatisticsDivisionList.do?pgmNo=179)에서 가져올 수 있습니다.
  - **문제 발생:** 이 출처에서 데이터를 가져올 때, 표 형식으로 처리할 수 없어서 오류나 불완전한 데이터가 발생하였습니다.
  - **해결 방안:** 이 문제를 해결하기 위해, Microsoft에 한글 지원 소프트웨어를 설치하고, 데이터를 쉽게 다운로드하고 변환할 수 있는 도구를 사용할 계획입니다.

- **공휴일 데이터:** 파일명: **holidays_data_korea_2024.csv**
  - **데이터 출처:** 네이버에서 공휴일 데이터를 찾을 수 있으며, 이를 수동으로 입력하여 파일로 저장할 수 있습니다.
  - **문제 발생:** 공휴일 정보 API는 로그인 후 사용할 수 있으므로, API를 통한 데이터 수집에 어려움이 있었습니다.
  - **해결 방안:** 일단 네이버에서 공휴일 데이터를 수동으로 다운로드하여 CSV 파일로 저장한 후, 추후 API에 로그인하여 자동으로 데이터를 가져올 수 있을 것입니다.

## 2. **데이터 분석 및 정제:**
  - 위의 출처들에서 데이터를 수집한 후, 분석을 위한 4주차(03/28)까지 데이터 전처리와 정제를 진행할 예정입니다. 이 단계에서는 데이터를 표준화하여 예측 모델에 사용할 수 있도록 준비할 것입니다.
