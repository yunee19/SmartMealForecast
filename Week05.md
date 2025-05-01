[주 5]

DONE:

최고 모델: Random Forest

모델 평가:

Linear Regression - MSE: 17866.6981, MAE: 99.7860

Random Forest - MSE: 16446.7781, MAE: 91.9047

XGBoost - MSE: 16536.3203, MAE: 89.8103

파일 test_model: 적합한 모델을 선택하고, 개선 및 구현 단계를 계속 진행 중입니다.

오차 (MSE, MAE):

현재 모델의 정확도를 평가하기 위해 오차 지표(MSE, MAE)를 계산했습니다.

One-hot Encoding을 적용하기 전에 입력 데이터를 개선했습니다.

입력 데이터 개선:

Feature Scaling: StandardScaler를 사용하여 입력 변수의 표준화를 실험하고 모델 성능을 개선했습니다.

새로운 특성 추가: 날씨 데이터, 공휴일 데이터(초복, 말복 등) 및 특별한 날을 merged_data.csv 파일에 추가하고 통합했습니다.

범주형 변수 인코딩: lunch_menu와 dinner_menu 열에서 lunch_rice, dinner_rice 등의 열로 변수를 인코딩했습니다. 그러나, 나중에 lunch_menu와 dinner_menu 열을 모두 인코딩하여 결과를 비교할 계획입니다.

고급 모델 실험:

Random Forest: 모델을 실험 중이며, 예측 정확도를 계속 점검해야 합니다.

One-hot Encoding과 Word Embedding:

One-hot Encoding: 모델 예측을 개선하기 위해 One-hot Encoding을 적용하고 있습니다.

Word Embedding: 아직 Word Embedding을 실험하지 않았습니다. 다음 주에 이 방법을 실험할 예정입니다.

TODO (다음 주 진행 사항):

Random Forest 모델의 정확도를 계속 점검하고 개선합니다.

One-hot Encoding과 Word Embedding 방법을 실험하고 결과를 비교합니다.

데이터가 완전히 인코딩되었는지 확인하고 결과를 다시 검토하여 모델이 정확하게 작동하는지 점검합니다
