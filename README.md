# Tomato Growth Stage Classification (EfficientNet-B0)

토마토의 생육단계 분류 모델 연구

## 프로젝트 개요
- **목표**: 토마토 5개 생육 단계(발아기, 유묘기, 영양생장기, 개화기, 착과/성숙기)를 분류하는 비전 분류 모델 구축
- **비교 실험**: Custom CNN, MobileNetV2, EfficientNet-B0 세 모델을 비교하여 EfficientNet-B0을 최종 선정
- **담당 역할**: 데이터 전처리 및 라벨링 관리, 모델 비교 실험 및 정량 평가

## 데이터
- 총 1,241장, Train/Val/Test = 6:2:2 분할
- 224x224 리사이즈, 정규화 mean/std = [0.5, 0.5, 0.5] (ImageNet 사전학습 미사용)
- Augmentation: RandomHorizontalFlip(0.5), RandomVerticalFlip(0.2), RandomRotation(20°), ColorJitter(0.3, 0.3, 0.3, 0.1)

## 모델 비교 결과

| 모델 | Params | Best Epoch | Val Acc | Test Macro F1 |
|---|---|---|---|---|
| Custom CNN | 12.85M | 52 | 83.33% | 0.8315 |
| MobileNetV2 | 2.23M | 84 | 95.56% | 0.8688 |
| **EfficientNet-B0 (최종)** | 4.01M | 66 | 95.79% | **0.8916** |

### EfficientNet-B0 클래스별 지표 (Test Set 257개)

| 생육 단계 | Precision | Recall | F1-Score |
|---|---|---|---|
| 발아기 (Germination) | 0.8727 | 0.9600 | 0.9143 |
| 유묘기 (Seeding) | 0.8235 | 0.8400 | 0.8317 |
| 영양생장기 (Vegetative) | 0.9259 | 0.8621 | 0.8929 |
| 개화기 (Flowering) | 0.9524 | 0.8163 | 0.8791 |
| 착과/성숙기 (Fruit_and_Ripening) | 0.9400 | 0.9400 | 0.9400 |
| **전체 평균 (Macro Avg)** | 0.9029 | 0.8837 | **0.8916** |

## 하이퍼파라미터
- Optimizer: AdamW (lr=1e-3, weight_decay=5e-4)
- Loss: BCEWithLogitsLoss
- Batch size: 64, Max epoch: 160, EarlyStopping patience: 20
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- 최적화: bfloat16 AMP, channels_last, torch.compile

## 트러블슈팅
초기 학습률(1e-3)과 강한 데이터 증강을 동시에 적용하면서 초반 Loss가 불안정했으나, ReduceLROnPlateau 스케줄러와 EarlyStopping(patience=20)으로 안정적인 수렴을 유도했습니다 (최저 검증 손실 0.1119, epoch 66).

## 실행 방법
```bash
python train_efficientnet.py
```
> `BASE_DIR` 경로는 본인 서버/로컬 환경에 맞게 수정 필요

## 참고 사항
- Validation Accuracy(95.79%)는 `BCEWithLogitsLoss` 기반 5개 클래스 각각을 원소 단위(element-wise)로 평가한 지표라 구조상 높게 집계됩니다. 실제 Top-1 기준 Test 정확도는 클래스별 Recall 기반으로 약 88% 수준입니다.
- 5개 클래스 단일 라벨 분류 문제이므로, CrossEntropyLoss가 이론적으로 더 적합한 손실 함수입니다. 이번 실험에서는 BCEWithLogitsLoss를 사용했고, CrossEntropy 전환은 향후 개선 과제로 남겨두었습니다.
