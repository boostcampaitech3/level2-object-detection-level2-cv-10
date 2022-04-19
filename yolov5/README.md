## Overview

YOLOv5  
**Yolov5l6, Yolov5x6** 사용

|Model|리더보드 mAP|
|---|:---:|
|YOLOv5l6|0.5468|
|YOLOv5x6_withTTA|0.590|
|YOLOv5x6_withTTA_modified|0.608|

## Install

  ```
  # pip install -r requirements.txt
  ```

## Usage

- coco data를 yolo data 형태로 변환하기 

coco_to_yol.py의 file_name을 원하는 대상 file로 바꾸고 실행합니다.


```bash
python coco_to_yolo.py
```

## Usage

- YOLOv5 모듈로 train 하기

train.py 를 실행, 내부 args 참고해서 hyperparameter 지정합니다. 

```bash
python train.py 
```

- YOLOv5 모듈로 validation 하기

위와 동일한 방법으로 validation을 진행합니다. 

내부 코드 수정을 통해 yolo format 출력 결과를 coco format 출력합니다.
수정된 코드는 다음과 같습니다. 

```bash

# ----- validation -----

    with open(file, 'a') as f:
        f.write((str(int(num)) + ','))
        for *xyxy, conf, cls in predn.tolist():
            line = (cls, conf ,*xyxy) if save_conf else (cls, *xyxy)
            #f.write(('%g ' * len(line)).rstrip() % line + ' |')
        #f.write(' ,'+'test/' + num + '.jpg' + '\n')
        #test/0009.jpg
  # ----------------------
```

- YOLOv5 모듈로 submission 하기

validation 을 통해 출력된 txt file의 순서를 정리하고 csv file 로 변환합니다. 
sorline.py의 file_name을 원하는 대상 file로 바꿔줍니다.

```bash

python sortline.py 

```




