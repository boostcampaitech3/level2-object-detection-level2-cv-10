# [Lv2 P-Stage] Object Detection :: #눈#사람
> 📑 Wrapup Report 보러가기 [>> PDF]()

## Members
| 김하준 | 송민수 | 심준교 | 유승리 | 이창진 | 전영우 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![눈사람_김하준](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![눈사람_송민수](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![눈사람_심준교](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![눈사람_유승리](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![눈사람_이창진](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![눈사람_전영우](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |
|[GitHub](https://github.com/HajunKim)|[GitHub](https://github.com/sooya233)|[GitHub](https://github.com/Shimjoonkyo)|[GitHub](https://github.com/seungriyou)|[GitHub](https://github.com/noisrucer)|[GitHub](https://github.com/wowo0709)|

<br>

## Competition : 재활용 품목 분류를 위한 Object Detection
<img width="1084" alt="스크린샷 2022-06-14 오후 1 27 45" src="https://user-images.githubusercontent.com/43572543/173493348-3feef421-a95c-44b9-97cd-664530bf2e93.png">

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

<br>

## Our Team's Contributions
```
- Cross Validation - Stratified Group Fold
- Ensemble 코드
- BiFPN 추가
- Universenet 추가
- YOLOv5
- EfficientDet
```
<br>

## Model Evaluation

### Cross-Validation Strategy

Validation mAP와 Public LB score의 mAP를 align시키기 위해 train set과 validation set이 비슷한 class 비율을 가지도록 scikit-learn에서 제공하는 `Stratified Group K Fold`를 사용하였다. 초반에 실험할때는 Fold 0으로 실험을 진행하였고 성능이 좋은 단일모델이 나오면 다른 fold에 적용을 하여 추후 앙상블 단계에서 큰 성능 향상을 가져다주었다. 또한 평균적으로 20 epochs 이내에 overfitting이 발생한다는 것을 관찰하고 validation set없이 전체 fold에 대해서 훈련을 진행하였고 public LB에서도 좋은 결과를 보였다.

### 1-stage  

YOLOX, YOLOv5, EfficientDet, Universenet, TOOD, ATSS 등 다양한 1-stage 모델들을 시도해 보았다. mmdetection 라이브러리에 제공되지 않는 모델들이 많아서 해당 repository에서 코드를 가져와서 훈련을 진행하였다.

- **YOLOX & YOLOv5** | YOLOX와 YOLOv5는 다른 모델들에 비해서 비교적 학습시간이 오래 걸렸지만 2-stage 모델들보다 LB score가 그렇게 좋지 않았다. 하지만 2-stage 모델들보다 비교적 small object AP가 높아 앙상블할때 모델 다양성에 기여를 했다. 

- **UniverseNet & ATSS** | UniverseNet과 ATSS는 backbone으로 Swin-L을 사용했을 때, heavy augmentation 적용 시 학습이 되지 않는 현상이 발생하였다. 따라서 이를 해결하기 위해서 neck(FPN)에서 `start_level`, `add_extra_convs`를 삭제하였다. 또한, 연산 속도를 높이기 위해 fp16을 적용하였다. 그리고 두 경우 모두 Cascade R-CNN보다 small과 medium을 더 잘 detect 하였지만, mAP50 결과는 좋지 않았다. 또한 ATSS 모델에서 추가적인 anchor box ratio를 추가해주었다.

- **EfficientDet** | EfficientDet은 backbone으로 EfficientNet, Neck으로 bifpn을 사용했다. 하지만 다른 모델들에 비해 좋은 성능을 보이지 못했다.

- **TOOD** | 대회 후반부에 TOOD모델이 작은 물체에 대한 성능이 매우 좋다고 들어서 적용해봤고 시도한 모든 모델중에 small과 medium object에 대한 AP가 가장 높았다.


### 2-stage
- **Backbone** | 주로 Swin Transformer를 사용했다. ResNet등 다른 backbone들을 시도해보았지만 트랜스포머 기반이 가장 성능이 우수했다. 

- **Neck** | FPN, PAFPN, BiFPN 등을 시도했지만 성능 차이는 크지 않았다. 

- **Detector** | Cascade RCNN, Faster RCNN 를 시도했는데 Cascade RCNN의 성능이 조금 더 좋았다. 

- **Etc** | EDA를 진행하며 object의 box ratio 분포가 다양하다는걸 관찰했고 기존의 베이스라인에 있던 anchor box ratio 0.5, 1, 2.0 외에 0.7과 1.5를 추가하였다. 또한 class imbalance문제를 해결하기 위해서 예측하기 힘든 물체에 더 포커스를 두는 focal loss를 Cascade R-CNN에 적용했고 그 결과 단일 모델 성능이 가장 좋았다.

### Heavy Augmentation
Classification task에 비해서 translation variance 문제가 중요한 object detection task에서는 모델의 capacity를 키워주는것이 중요하다고 판단했다. 또한 Kaggle등 대회 수상자들의 discussion을 살펴보았을때 heavy augmentation이 좋은 결과를 가져다 준다는것을 알았고 이번 대회에서 heavy augmentation을 적용한 결과 성능 향상을 가져다주었다. Albumentation 라이브러리를 사용하였고 공통적으로 사용한 augmentation 기법은 아래와 같다.

```
1. ShiftScaleRotate
2. RandomBrightnessContrast
3. RGBShift
4. HueSaturationValue
5. JpegCompression
6. ChannelShuffle
7. MedianBlur
8. CLAHE
```

<br>

## Ensemble    
NMS와 Soft-NMS같이 redundant box를 제거하는 앙상블 기법보다 모든 예측 박스를 사용하는 WBF 알고리즘이 가장 좋은 성능을 보여주었고 단일모델보다 `6 mAP` 가량 높은 점수를 달성 하였다. 다양한 `iou_threshold`와 `score_threshold`를 바꿔가며 실험을 진행하였고 결과적으로 `iou_threshold=0.05`, `score_threshold=0.01`이 가장 좋은 성능을 보여주었다. 중복 박스를 제거하는 NMS와 Soft-NMS와 달리 WBF는 예측 박스들의 분포와 parameter값들에 따라 결과가 상당히 민감하게 반응 하였다. 하지만 다른 앙상블 기법에 비해 parameter와 성능의 관계를 파악하기가 어려웠다. 예를들어 `score_threshold=0.01`은 confidence score이 0.01 이하인 예측값들을 지워주는데 confidence score이 0.01이면 상당히 퀄리티가 낮은 예측값이라고 생각해 `score_threshold`를 높여줬지만 성능이 하락하였다. 최종적으로 40개 이상의 단일 모델을 앙상블 하였고 그 결과 팀 최고점수를 달성하였다.

<br>

## Further Improvements   
여러가지 모델들을 사용하여 실험을 진행해 봤지만, 다른 팀과 비교해 보았을 때, 같은 epoch임에도 불구하고 성능 차이가 최대 5%정도 크게 벌어졌다. 이는 cross validation 진행시 fold의 문제이거나, 실험을 진행할 시에 seed를 1333로 고정시켜 줬었는데, 해당과정들에서 다른 팀보다 덜 잘 분포된 training set이 만들어져서 점수가 낮게 나오지 않았나 생각한다.

Yolo v5 가 상당한 성능이 나왔음에도 앙상블 결과 큰 성능향상이 없었다. 1,2 등 팀은 비교적 적은 모델로 앙상블을 진행했다는 것(Yolov5, Swin-Cascade 기반 모델)을 고려할 때 너무 많은 모델로 앙상블을 한게 오히려 성능에 악영향을 끼쳤을 수도 있을 것 같다.

다른 팀들보다 단일모델의 성능이 떨어지는 경향이 있었다. 점수가 정체됐을때 데이터로 돌아가서 EDA를 진행하면서 인사이트를 얻어서 다른 방식으로 접근 했었으면 더 좋았을 것 같다.

또한 Leader board와 잘 align되는 validation set을 찾는 것이 중요하다. Stratified group k-fold에서 class 분포 뿐만 아니라 bbox ratio, bbox area, bbox count 등으로 다양하게 실험했으면 더 좋았을 것 같다.

WBF 앙상블에서 주로 `iou_threshold`와 `score_threshold`를 바꿔주면서 실험을 진행하였는데 다음 대회에서는 random seed, snapshot ensemble, SWA등 조금 더 다양한 앙상블 기법들을 시도해 볼 것이다. 

Box ratio 와 area를 기반으로 outlier 를 제거했었는데 오히려 성능이 떨어졌었다. 해당 사유를 정확하게 파악하지 못해서 아쉽다.

<br>

## Experiments
| Index | Property | Name | LB Score |
| --- | --- | --- | --- |
| 1 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold0 / inf 800 | 0.6177 |
| 2 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold0 / inf 1024 | 0.6205 |
| 3 | Ensemble | 01 / 02 / inf 800, 1024 // nms 0.55 | 0.6423 |
| 4 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold1 / Train 1024 / inf 1024 | 0.6157 |
| 5 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold0 / Train 800 / inf 800 | 0.607 |
| 6 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold0 / Train 800 / inf 1024 | 0.6104 |
| 7 | Ensemble | 01 / 02 / 05 / 06 // nms 055 | 0.6472 |
| 8 | Ensemble | 01 / 02 / 05 / 06 / YoloX / WBF 0.55 skip 0.1 | 0.6324 |
| 9 | Ensemble | 01 / 02 / 05 / 06 / YoloX / nms 0.55 | 0.6361 |
| 10 | Ensemble | 14개 모델 / WBF 055, skip 0.1 | 0.6813 |
| 11 | Ensemble | 16개 모델 / WBF 05, skip 0.08 | 0.685 |
| 12 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold1 / inf 1024 | - |
| 13 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold2 / inf 1024 | - |
| 14 | Ensemble | 01 / 02 / 12-1 / 12-2 / 13 // nms 055 | 0.6539 |
| 15 | Single-2stage | Faster_RCNN / Swin_L / FPN / fold0 / inf 512 | 0.5843 |
| 16 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 1024 | 0.6241 |
| 17 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 800 | - |
| 18 | Ensemble | 22개 모델 / WBF 0.5, skip 0.1 | 0.6894 |
| 19 | Ensemble | 23개 모델 / WBF 0.55, skip 0.1 | 0.6914 |
| 20 | Ensemble | 23개 모델 / WBF 0.6, skip 0.1 | 0.6904 |
| 21 | Ensemble | 23개 모델 / WBF 0.4, skip 0.1 | 0.6778 |
| 22 | Ensemble | 23개 모델 / WBF 0.55, skip 0.08 | 0.6931 |
| 23 | Ensemble | 23개 모델 / WBF 0.55, skip 0.06 | 0.6945 |
| 24 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf1024 + outlier 제거  | 0.5962 |
| 25 | Ensemble | Multi-stage Ensemble | 0.676 |
| 26 | Ensemble | 25개 모델 / WBF 0.55, skip 0.1 / model_weights | 0.6931 |
| 27 | Ensemble | 25개 모델 / WBF 0.55, skip 0.06 | - |
| 28 | Ensemble | 27개 모델 / WBF 0.55, skip 0.05 | - |
| 29 | Ensemble | 27개 모델 / WBF 0.55, skip 0.05 / model_weights | 0.6984 |
| 30 | Single-1stage | Yolov5 | 0.5468 |
| 31 | Single-1stage | Yolov5 | 0.5907 |
| 32 | Single-1stage | ATSS / Swin-L / FPN | 0.5563 |
| 33 | Single-2stage | Cascade R-CNN / Swin-L / FPN / bbox head / full | 0.6246 |
| 34 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / fold0 | 0.6243 |
| 35 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / full | 0.6368 |
| 36 | Single-1stage | UniverseNet / Swin-L / FPN+SEPC / fold0 | 0.5684 |
