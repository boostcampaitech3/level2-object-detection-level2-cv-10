# Level2 P-stage: Object Detection #눈#사람
<img src="https://user-images.githubusercontent.com/71958885/164018511-7474616a-e5c4-421e-b28a-5b35fb43b726.png" width="200" height="200" />

<br>

## Members
* 김하준 T3066
* 송민수 T3113
* 심준교 T3124
* 유승리 T3129
* 이창진 T3169
* 전영우 T3192
<br>

## Mentor
* 강천성 멘토님
<br>

## Compatition : 재활용 품목 분류를 위한 Object Detection
바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

![image](https://user-images.githubusercontent.com/71958885/164008385-32e7e2c7-e8d3-4661-bcc5-9775910a21a8.png)

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎
<br>

## Our team's contribution
```
- Cross Validation - Stratified Goup Fold
- Ensemble 코드
- BiFPN 추가
- Universenet 추가
```
<br>

## Experiments
| Index | Property | Name | LB Score | Submitter | Date |
| --- | --- | --- | --- | --- | --- |
| 1 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold0 / inf 800 | 0.6177 | 유승리 캠퍼 | 2022/03/29 |
| 2 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold0 / inf 1024 | 0.6205 | 유승리 캠퍼 | 2022/03/29 |
| 3 | Ensemble | 01 / 02 / inf 800, 1024 // nms 0.55 | 0.6423 | 이창진 캠퍼 | 2022/03/29 |
| 4 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold1 / Train 1024 / inf 1024 | 0.6157 | 이창진 캠퍼 | 2022/03/30 |
| 5 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold0 / Train 800 / inf 800 | 0.607 | 유승리 캠퍼 | 2022/03/30 |
| 6 | Single-2stage | Cascade R-CNN / Swin L / PAFPN / fold0 / Train 800 / inf 1024 | 0.6104 | 유승리 캠퍼 | 2022/03/30 |
| 7 | Ensemble | 01 / 02 / 05 / 06 // nms 055 | 0.6472 | 유승리 캠퍼 | 2022/03/30 |
| 8 | Ensemble | 01 / 02 / 05 / 06 / YoloX / WBF 0.55 skip 0.1 | 0.6324 |  | 2022/03/30 |
| 9 | Ensemble | 01 / 02 / 05 / 06 / YoloX / nms 0.55 | 0.6361 |  | 2022/03/30 |
| 10 | Ensemble | 14개 모델 / WBF 055, skip 0.1 | 0.6813 | 이창진 캠퍼 | 2022/03/31 |
| 11 | Ensemble | 16개 모델 / WBF 05, skip 0.08 | 0.685 | 이창진 캠퍼 | 2022/03/31 |
| 12 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold1 / inf 1024 |  |  | 2022/03/31 |
| 13 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold2 / inf 1024 |  |  | 2022/03/31 |
| 14 | Ensemble | 01 / 02 / 12-1 / 12-2 / 13 // nms 055 | 0.6539 | 유승리 캠퍼 | 2022/03/31 |
| 15 | Single-2stage | Faster_RCNN / Swin_L / FPN / fold0 / inf 512 | 0.5843 | 심준교 캠퍼 | 2022/03/28 |
| 16 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 1024 | 0.6241 | 심준교 캠퍼 | 2022/04/01 |
| 17 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 800 |  | 심준교 캠퍼 | 2022/04/01 |
| 18 | Ensemble | 22개 모델 / WBF 0.5, skip 0.1 | 0.6894 | 이창진 캠퍼 | 2022/04/01 |
| 19 | Ensemble | 23개 모델 / WBF 0.55, skip 0.1 | 0.6914 | 이창진 캠퍼 | 2022/04/01 |
| 20 | Ensemble | 23개 모델 / WBF 0.6, skip 0.1 | 0.6904 | 이창진 캠퍼 | 2022/04/01 |
| 21 | Ensemble | 23개 모델 / WBF 0.4, skip 0.1 | 0.6778 | 이창진 캠퍼 | 2022/04/01 |
| 22 | Ensemble | 23개 모델 / WBF 0.55, skip 0.08 | 0.6931 | 이창진 캠퍼 | 2022/04/01 |
| 23 | Ensemble | 23개 모델 / WBF 0.55, skip 0.06 | 0.6945 | 이창진 캠퍼 | 2022/04/01 |
| 24 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf1024 + outlier 제거  | 0.5962 | 이창진 캠퍼 | 2022/04/02 |
| 25 | Ensemble | Multi-stage Ensemble | 0.676 | 이창진 캠퍼 | 2022/04/02 |
| 26 | Ensemble | 25개 모델 / WBF 0.55, skip 0.1 / model_weights | 0.6931 | 이창진 캠퍼 | 2022/04/02 |
| 27 | Ensemble | 25개 모델 / WBF 0.55, skip 0.06 |  |  | 2022/04/02 |
| 28 | Ensemble | 27개 모델 / WBF 0.55, skip 0.05 |  |  | 2022/04/02 |
| 29 | Ensemble | 27개 모델 / WBF 0.55, skip 0.05 / model_weights | 0.6984 | 이창진 캠퍼 | 2022/04/02 |
| 30 | Single-1stage | Yolov5 | 0.5468 | 김하준 캠퍼 | 2022/04/04 |
| 31 | Single-1stage | Yolov5 | 0.5907 | 김하준 캠퍼 | 2022/04/05 |
| 32 | Single-1stage | ATSS / Swin-L / FPN | 0.5563 | 유승리 캠퍼 | 2022/03/28 |
| 33 | Single-2stage | Cascade R-CNN / Swin-L / FPN / bbox head / full | 0.6246 | 유승리 캠퍼 | 2022/04/04 |
| 34 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / fold0 | 0.6243 | 유승리 캠퍼 | 2022/04/04 |
| 35 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / full | 0.6368 | 유승리 캠퍼 | 2022/04/04 |
| 36 | Single-1stage | UniverseNet / Swin-L / FPN+SEPC / fold0 | 0.5684 | 유승리 캠퍼 | 2022/04/05 |
