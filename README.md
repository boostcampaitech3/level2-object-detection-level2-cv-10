# [Lv2 P-Stage] Object Detection / #๋#์ฌ๋
> ๐ Wrapup Report [โบ PDF](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-10/blob/main/Object%20Detection_CV_%E1%84%90%E1%85%B5%E1%86%B7%20%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(10%E1%84%8C%E1%85%A9).pdf)

## Members
| ๊นํ์ค | ์ก๋ฏผ์ | ์ฌ์ค๊ต | ์ ์น๋ฆฌ | ์ด์ฐฝ์ง | ์ ์์ฐ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![แแฎแซแแกแแกแท_แแตแทแแกแแฎแซ](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![แแฎแซแแกแแกแท_แแฉแผแแตแซแแฎ](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![แแฎแซแแกแแกแท_แแตแทแแฎแซแแญ](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![แแฎแซแแกแแกแท_แแฒแแณแผแแต](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![แแฎแซแแกแแกแท_แแตแแกแผแแตแซ](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![แแฎแซแแกแแกแท_แแฅแซแแงแผแแฎ](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |
|[GitHub](https://github.com/HajunKim)|[GitHub](https://github.com/sooya233)|[GitHub](https://github.com/Shimjoonkyo)|[GitHub](https://github.com/seungriyou)|[GitHub](https://github.com/noisrucer)|[GitHub](https://github.com/wowo0709)|

<br>

## Competition : ์ฌํ์ฉ ํ๋ชฉ ๋ถ๋ฅ๋ฅผ ์ํ Object Detection
<img width="1084" alt="แแณแแณแแตแซแแฃแบ 2022-06-14 แแฉแแฎ 1 27 45" src="https://user-images.githubusercontent.com/43572543/173493348-3feef421-a95c-44b9-97cd-664530bf2e93.png">

### Introduction

๋ฐ์ผํ๋ก ๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ , ์๋น๋๋ ์๋๋ฅผ ์ด๊ณ  ์์ต๋๋ค. ํ์ง๋ง ์ด๋ฌํ ๋ฌธํ๋ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์์ต๋๋ค.

๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค.

๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Detection ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก๋ ์ผ๋ฐ ์ฐ๋ ๊ธฐ, ํ๋ผ์คํฑ, ์ข์ด, ์ ๋ฆฌ ๋ฑ 10 ์ข๋ฅ์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋ฉ๋๋ค.

์ฌ๋ฌ๋ถ์ ์ํด ๋ง๋ค์ด์ง ์ฐ์ํ ์ฑ๋ฅ์ ๋ชจ๋ธ์ ์ฐ๋ ๊ธฐ์ฅ์ ์ค์น๋์ด ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ๋๊ฑฐ๋, ์ด๋ฆฐ์์ด๋ค์ ๋ถ๋ฆฌ์๊ฑฐ ๊ต์ก ๋ฑ์ ์ฌ์ฉ๋  ์ ์์ ๊ฒ์๋๋ค. ๋ถ๋ ์ง๊ตฌ๋ฅผ ์๊ธฐ๋ก๋ถํฐ ๊ตฌํด์ฃผ์ธ์! ๐

### Metric

Test set์ mAP50(Mean Average Precision)

<br>

## Main Contributions
```
- Cross Validation - Stratified Group Fold
- Ensemble ์ฝ๋
- BiFPN ์ถ๊ฐ
- Universenet ์ถ๊ฐ
- YOLOv5
- EfficientDet
```

### MMDetection
[>> ๊ฐ์ธ๋ณ config ํด๋](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-10/tree/main/mmdetection/configs)

[>> UniverseNet ์ถ๊ฐ](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-10/tree/main/mmdetection/mmdet/models)

### YOLOv5
[>> YOLOv5 ํด๋](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-10/tree/main/yolov5)

<br>

## Model Evaluation

### Cross-Validation Strategy

Validation mAP์ Public LB score์ mAP๋ฅผ align์ํค๊ธฐ ์ํด train set๊ณผ validation set์ด ๋น์ทํ class ๋น์จ์ ๊ฐ์ง๋๋ก scikit-learn์์ ์ ๊ณตํ๋ `Stratified Group K Fold`๋ฅผ ์ฌ์ฉํ์๋ค. ์ด๋ฐ์ ์คํํ ๋๋ Fold 0์ผ๋ก ์คํ์ ์งํํ์๊ณ  ์ฑ๋ฅ์ด ์ข์ ๋จ์ผ๋ชจ๋ธ์ด ๋์ค๋ฉด ๋ค๋ฅธ fold์ ์ ์ฉ์ ํ์ฌ ์ถํ ์์๋ธ ๋จ๊ณ์์ ํฐ ์ฑ๋ฅ ํฅ์์ ๊ฐ์ ธ๋ค์ฃผ์๋ค. ๋ํ ํ๊ท ์ ์ผ๋ก 20 epochs ์ด๋ด์ overfitting์ด ๋ฐ์ํ๋ค๋ ๊ฒ์ ๊ด์ฐฐํ๊ณ  validation set์์ด ์ ์ฒด fold์ ๋ํด์ ํ๋ จ์ ์งํํ์๊ณ  public LB์์๋ ์ข์ ๊ฒฐ๊ณผ๋ฅผ ๋ณด์๋ค.

### 1-Stage  
YOLOX, YOLOv5, EfficientDet, Universenet, TOOD, ATSS ๋ฑ ๋ค์ํ 1-stage ๋ชจ๋ธ๋ค์ ์๋ํด ๋ณด์๋ค. mmdetection ๋ผ์ด๋ธ๋ฌ๋ฆฌ์ ์ ๊ณต๋์ง ์๋ ๋ชจ๋ธ๋ค์ด ๋ง์์ ํด๋น repository์์ ์ฝ๋๋ฅผ ๊ฐ์ ธ์์ ํ๋ จ์ ์งํํ์๋ค.

  | Detector     | Backbone        | Neck       | Optimizer | Scheduler       | ํน์ด์ฌํญ                   |
  |--------------|-----------------|------------|-----------|-----------------|----------------------------|
  | Universenet  | Res2Net         | FPN, SEPC  | SGD       | StepLR          |                            |
  | Yolo X       | CSP-DarkNet     | YOLOXPAFPN | SGD       | stepLR          | AnchorFree                 |
  | Yolo v5(xl6) | CSP-DarkNet     | SPPF       | SGD       | Lambda LR       | 1536X1536   inference, TTA |
  | ATSS         | Swin-L          | FPN        | AdamW     | StepLR          | fp16                       |
  | UniverseNet  | Swin-L          | FPN, SEPC  | SGD       | StepLR          | fp16, grad_clip            |
  | TOOD         | ResNeXt, Swin-L | FPN        | AdamW     | CosineAnnealing |                            |

- **YOLOX & YOLOv5** | YOLOX์ YOLOv5๋ ๋ค๋ฅธ ๋ชจ๋ธ๋ค์ ๋นํด์ ๋น๊ต์  ํ์ต์๊ฐ์ด ์ค๋ ๊ฑธ๋ ธ์ง๋ง 2-stage ๋ชจ๋ธ๋ค๋ณด๋ค LB score๊ฐ ๊ทธ๋ ๊ฒ ์ข์ง ์์๋ค. ํ์ง๋ง 2-stage ๋ชจ๋ธ๋ค๋ณด๋ค ๋น๊ต์  small object AP๊ฐ ๋์ ์์๋ธํ ๋ ๋ชจ๋ธ ๋ค์์ฑ์ ๊ธฐ์ฌ๋ฅผ ํ๋ค. 

- **UniverseNet & ATSS** | UniverseNet๊ณผ ATSS๋ backbone์ผ๋ก Swin-L์ ์ฌ์ฉํ์ ๋, heavy augmentation ์ ์ฉ ์ ํ์ต์ด ๋์ง ์๋ ํ์์ด ๋ฐ์ํ์๋ค. ๋ฐ๋ผ์ ์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด์ neck(FPN)์์ `start_level`, `add_extra_convs`๋ฅผ ์ญ์ ํ์๋ค. ๋ํ, ์ฐ์ฐ ์๋๋ฅผ ๋์ด๊ธฐ ์ํด fp16์ ์ ์ฉํ์๋ค. ๊ทธ๋ฆฌ๊ณ  ๋ ๊ฒฝ์ฐ ๋ชจ๋ Cascade R-CNN๋ณด๋ค small๊ณผ medium์ ๋ ์ detect ํ์์ง๋ง, mAP50 ๊ฒฐ๊ณผ๋ ์ข์ง ์์๋ค. ๋ํ ATSS ๋ชจ๋ธ์์ ์ถ๊ฐ์ ์ธ anchor box ratio๋ฅผ ์ถ๊ฐํด์ฃผ์๋ค.

- **EfficientDet** | EfficientDet์ backbone์ผ๋ก EfficientNet, Neck์ผ๋ก bifpn์ ์ฌ์ฉํ๋ค. ํ์ง๋ง ๋ค๋ฅธ ๋ชจ๋ธ๋ค์ ๋นํด ์ข์ ์ฑ๋ฅ์ ๋ณด์ด์ง ๋ชปํ๋ค.

- **TOOD** | ๋ํ ํ๋ฐ๋ถ์ TOOD๋ชจ๋ธ์ด ์์ ๋ฌผ์ฒด์ ๋ํ ์ฑ๋ฅ์ด ๋งค์ฐ ์ข๋ค๊ณ  ๋ค์ด์ ์ ์ฉํด๋ดค๊ณ  ์๋ํ ๋ชจ๋  ๋ชจ๋ธ์ค์ small๊ณผ medium object์ ๋ํ AP๊ฐ ๊ฐ์ฅ ๋์๋ค.


### 2-Stage
  |     Detector                     |     Backbone           |     Neck                     |     Optimizer    |     Scheduler                    |     ํน์ด์ฌํญ                                |
  |----------------------------------|------------------------|------------------------------|------------------|----------------------------------|---------------------------------------------|
  |     Faster-RCNN, Cascade-RCNN    |     Swin-L(224,384)    |     FPN, PAFPN, FPN_carafe    |     AdamW        |     StepLR, CosineAnnealingLR     |                                             |
  |     Cascade R-CNN                |     Swin-L             |     FPN, PAFPN, BiFPN        |     AdamW        |     StepLR, CosineAnnealingLR    |     Focal loss,   Shared4Conv1FCBBoxHead    |

- **Backbone** | ์ฃผ๋ก Swin Transformer๋ฅผ ์ฌ์ฉํ๋ค. ResNet๋ฑ ๋ค๋ฅธ backbone๋ค์ ์๋ํด๋ณด์์ง๋ง ํธ๋์คํฌ๋จธ ๊ธฐ๋ฐ์ด ๊ฐ์ฅ ์ฑ๋ฅ์ด ์ฐ์ํ๋ค. 

- **Neck** | FPN, PAFPN, BiFPN ๋ฑ์ ์๋ํ์ง๋ง ์ฑ๋ฅ ์ฐจ์ด๋ ํฌ์ง ์์๋ค. 

- **Detector** | Cascade RCNN, Faster RCNN ๋ฅผ ์๋ํ๋๋ฐ Cascade RCNN์ ์ฑ๋ฅ์ด ์กฐ๊ธ ๋ ์ข์๋ค. 

- **Etc** | EDA๋ฅผ ์งํํ๋ฉฐ object์ box ratio ๋ถํฌ๊ฐ ๋ค์ํ๋ค๋๊ฑธ ๊ด์ฐฐํ๊ณ  ๊ธฐ์กด์ ๋ฒ ์ด์ค๋ผ์ธ์ ์๋ anchor box ratio 0.5, 1, 2.0 ์ธ์ 0.7๊ณผ 1.5๋ฅผ ์ถ๊ฐํ์๋ค. ๋ํ class imbalance๋ฌธ์ ๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด์ ์์ธกํ๊ธฐ ํ๋  ๋ฌผ์ฒด์ ๋ ํฌ์ปค์ค๋ฅผ ๋๋ focal loss๋ฅผ Cascade R-CNN์ ์ ์ฉํ๊ณ  ๊ทธ ๊ฒฐ๊ณผ ๋จ์ผ ๋ชจ๋ธ ์ฑ๋ฅ์ด ๊ฐ์ฅ ์ข์๋ค.

### Heavy Augmentation
Classification task์ ๋นํด์ translation variance ๋ฌธ์ ๊ฐ ์ค์ํ object detection task์์๋ ๋ชจ๋ธ์ capacity๋ฅผ ํค์์ฃผ๋๊ฒ์ด ์ค์ํ๋ค๊ณ  ํ๋จํ๋ค. ๋ํ Kaggle๋ฑ ๋ํ ์์์๋ค์ discussion์ ์ดํด๋ณด์์๋ heavy augmentation์ด ์ข์ ๊ฒฐ๊ณผ๋ฅผ ๊ฐ์ ธ๋ค ์ค๋ค๋๊ฒ์ ์์๊ณ  ์ด๋ฒ ๋ํ์์ heavy augmentation์ ์ ์ฉํ ๊ฒฐ๊ณผ ์ฑ๋ฅ ํฅ์์ ๊ฐ์ ธ๋ค์ฃผ์๋ค. Albumentation ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ฌ์ฉํ์๊ณ  ๊ณตํต์ ์ผ๋ก ์ฌ์ฉํ augmentation ๊ธฐ๋ฒ์ ์๋์ ๊ฐ๋ค.

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
NMS์ Soft-NMS๊ฐ์ด redundant box๋ฅผ ์ ๊ฑฐํ๋ ์์๋ธ ๊ธฐ๋ฒ๋ณด๋ค ๋ชจ๋  ์์ธก ๋ฐ์ค๋ฅผ ์ฌ์ฉํ๋ WBF ์๊ณ ๋ฆฌ์ฆ์ด ๊ฐ์ฅ ์ข์ ์ฑ๋ฅ์ ๋ณด์ฌ์ฃผ์๊ณ  ๋จ์ผ๋ชจ๋ธ๋ณด๋ค `6 mAP` ๊ฐ๋ ๋์ ์ ์๋ฅผ ๋ฌ์ฑ ํ์๋ค. ๋ค์ํ `iou_threshold`์ `score_threshold`๋ฅผ ๋ฐ๊ฟ๊ฐ๋ฉฐ ์คํ์ ์งํํ์๊ณ  ๊ฒฐ๊ณผ์ ์ผ๋ก `iou_threshold=0.05`, `score_threshold=0.01`์ด ๊ฐ์ฅ ์ข์ ์ฑ๋ฅ์ ๋ณด์ฌ์ฃผ์๋ค. ์ค๋ณต ๋ฐ์ค๋ฅผ ์ ๊ฑฐํ๋ NMS์ Soft-NMS์ ๋ฌ๋ฆฌ WBF๋ ์์ธก ๋ฐ์ค๋ค์ ๋ถํฌ์ parameter๊ฐ๋ค์ ๋ฐ๋ผ ๊ฒฐ๊ณผ๊ฐ ์๋นํ ๋ฏผ๊ฐํ๊ฒ ๋ฐ์ ํ์๋ค. ํ์ง๋ง ๋ค๋ฅธ ์์๋ธ ๊ธฐ๋ฒ์ ๋นํด parameter์ ์ฑ๋ฅ์ ๊ด๊ณ๋ฅผ ํ์ํ๊ธฐ๊ฐ ์ด๋ ค์ ๋ค. ์๋ฅผ๋ค์ด `score_threshold=0.01`์ confidence score์ด 0.01 ์ดํ์ธ ์์ธก๊ฐ๋ค์ ์ง์์ฃผ๋๋ฐ confidence score์ด 0.01์ด๋ฉด ์๋นํ ํ๋ฆฌํฐ๊ฐ ๋ฎ์ ์์ธก๊ฐ์ด๋ผ๊ณ  ์๊ฐํด `score_threshold`๋ฅผ ๋์ฌ์คฌ์ง๋ง ์ฑ๋ฅ์ด ํ๋ฝํ์๋ค. ์ต์ข์ ์ผ๋ก 40๊ฐ ์ด์์ ๋จ์ผ ๋ชจ๋ธ์ ์์๋ธ ํ์๊ณ  ๊ทธ ๊ฒฐ๊ณผ ํ ์ต๊ณ ์ ์๋ฅผ ๋ฌ์ฑํ์๋ค.

<br>

## Further Improvements   
์ฌ๋ฌ๊ฐ์ง ๋ชจ๋ธ๋ค์ ์ฌ์ฉํ์ฌ ์คํ์ ์งํํด ๋ดค์ง๋ง, ๋ค๋ฅธ ํ๊ณผ ๋น๊ตํด ๋ณด์์ ๋, ๊ฐ์ epoch์์๋ ๋ถ๊ตฌํ๊ณ  ์ฑ๋ฅ ์ฐจ์ด๊ฐ ์ต๋ 5%์ ๋ ํฌ๊ฒ ๋ฒ์ด์ก๋ค. ์ด๋ cross validation ์งํ์ fold์ ๋ฌธ์ ์ด๊ฑฐ๋, ์คํ์ ์งํํ  ์์ seed๋ฅผ 1333๋ก ๊ณ ์ ์์ผ ์คฌ์๋๋ฐ, ํด๋น๊ณผ์ ๋ค์์ ๋ค๋ฅธ ํ๋ณด๋ค ๋ ์ ๋ถํฌ๋ training set์ด ๋ง๋ค์ด์ ธ์ ์ ์๊ฐ ๋ฎ๊ฒ ๋์ค์ง ์์๋ ์๊ฐํ๋ค.

Yolo v5 ๊ฐ ์๋นํ ์ฑ๋ฅ์ด ๋์์์๋ ์์๋ธ ๊ฒฐ๊ณผ ํฐ ์ฑ๋ฅํฅ์์ด ์์๋ค. 1,2 ๋ฑ ํ์ ๋น๊ต์  ์ ์ ๋ชจ๋ธ๋ก ์์๋ธ์ ์งํํ๋ค๋ ๊ฒ(Yolov5, Swin-Cascade ๊ธฐ๋ฐ ๋ชจ๋ธ)์ ๊ณ ๋ คํ  ๋ ๋๋ฌด ๋ง์ ๋ชจ๋ธ๋ก ์์๋ธ์ ํ๊ฒ ์คํ๋ ค ์ฑ๋ฅ์ ์์ํฅ์ ๋ผ์ณค์ ์๋ ์์ ๊ฒ ๊ฐ๋ค.

๋ค๋ฅธ ํ๋ค๋ณด๋ค ๋จ์ผ๋ชจ๋ธ์ ์ฑ๋ฅ์ด ๋จ์ด์ง๋ ๊ฒฝํฅ์ด ์์๋ค. ์ ์๊ฐ ์ ์ฒด๋์๋ ๋ฐ์ดํฐ๋ก ๋์๊ฐ์ EDA๋ฅผ ์งํํ๋ฉด์ ์ธ์ฌ์ดํธ๋ฅผ ์ป์ด์ ๋ค๋ฅธ ๋ฐฉ์์ผ๋ก ์ ๊ทผ ํ์์ผ๋ฉด ๋ ์ข์์ ๊ฒ ๊ฐ๋ค.

๋ํ Leader board์ ์ align๋๋ validation set์ ์ฐพ๋ ๊ฒ์ด ์ค์ํ๋ค. Stratified group k-fold์์ class ๋ถํฌ ๋ฟ๋ง ์๋๋ผ bbox ratio, bbox area, bbox count ๋ฑ์ผ๋ก ๋ค์ํ๊ฒ ์คํํ์ผ๋ฉด ๋ ์ข์์ ๊ฒ ๊ฐ๋ค.

WBF ์์๋ธ์์ ์ฃผ๋ก `iou_threshold`์ `score_threshold`๋ฅผ ๋ฐ๊ฟ์ฃผ๋ฉด์ ์คํ์ ์งํํ์๋๋ฐ ๋ค์ ๋ํ์์๋ random seed, snapshot ensemble, SWA๋ฑ ์กฐ๊ธ ๋ ๋ค์ํ ์์๋ธ ๊ธฐ๋ฒ๋ค์ ์๋ํด ๋ณผ ๊ฒ์ด๋ค. 

Box ratio ์ area๋ฅผ ๊ธฐ๋ฐ์ผ๋ก outlier ๋ฅผ ์ ๊ฑฐํ์๋๋ฐ ์คํ๋ ค ์ฑ๋ฅ์ด ๋จ์ด์ก์๋ค. ํด๋น ์ฌ์ ๋ฅผ ์ ํํ๊ฒ ํ์ํ์ง ๋ชปํด์ ์์ฝ๋ค.

<br>

## LB Score Chart

<img width="369" alt="image" src="https://user-images.githubusercontent.com/43572543/173495756-488f65b3-cfd0-43c7-9029-8386c7cbacb2.png">

### Final LB Score
- **[Public]** mAP: 0.7026 (8์) 
- **[Private]** mAP: 0.6881 (8์)


<br>

## Experiments

<details>
<summary>LB ์ ์ถ ๊ธฐ๋ก</summary>
<div markdown="1">
  
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
  | 10 | Ensemble | 14๊ฐ ๋ชจ๋ธ / WBF 055, skip 0.1 | 0.6813 |
  | 11 | Ensemble | 16๊ฐ ๋ชจ๋ธ / WBF 05, skip 0.08 | 0.685 |
  | 12 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold1 / inf 1024 | - |
  | 13 | Single-2stage | Cascade R-CNN / Swin L / FPN / fold2 / inf 1024 | - |
  | 14 | Ensemble | 01 / 02 / 12-1 / 12-2 / 13 // nms 055 | 0.6539 |
  | 15 | Single-2stage | Faster_RCNN / Swin_L / FPN / fold0 / inf 512 | 0.5843 |
  | 16 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 1024 | 0.6241 |
  | 17 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf 800 | - |
  | 18 | Ensemble | 22๊ฐ ๋ชจ๋ธ / WBF 0.5, skip 0.1 | 0.6894 |
  | 19 | Ensemble | 23๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.1 | 0.6914 |
  | 20 | Ensemble | 23๊ฐ ๋ชจ๋ธ / WBF 0.6, skip 0.1 | 0.6904 |
  | 21 | Ensemble | 23๊ฐ ๋ชจ๋ธ / WBF 0.4, skip 0.1 | 0.6778 |
  | 22 | Ensemble | 23๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.08 | 0.6931 |
  | 23 | Ensemble | 23๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.06 | 0.6945 |
  | 24 | Single-2stage | Faster_RCNN / Swin_L / PAFPN / fold0 / inf1024 + outlier ์ ๊ฑฐ  | 0.5962 |
  | 25 | Ensemble | Multi-stage Ensemble | 0.676 |
  | 26 | Ensemble | 25๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.1 / model_weights | 0.6931 |
  | 27 | Ensemble | 25๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.06 | - |
  | 28 | Ensemble | 27๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.05 | - |
  | 29 | Ensemble | 27๊ฐ ๋ชจ๋ธ / WBF 0.55, skip 0.05 / model_weights | 0.6984 |
  | 30 | Single-1stage | Yolov5 | 0.5468 |
  | 31 | Single-1stage | Yolov5 | 0.5907 |
  | 32 | Single-1stage | ATSS / Swin-L / FPN | 0.5563 |
  | 33 | Single-2stage | Cascade R-CNN / Swin-L / FPN / bbox head / full | 0.6246 |
  | 34 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / fold0 | 0.6243 |
  | 35 | Single-2stage | Cascade R-CNN / Swin-L / FPN / focal loss / full | 0.6368 |
  | 36 | Single-1stage | UniverseNet / Swin-L / FPN+SEPC / fold0 | 0.5684 |

</div>
</details>
