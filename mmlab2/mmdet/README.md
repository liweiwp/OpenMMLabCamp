#### 气球目标检测
##### Colab教程

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liweiwp/OpenMMLabCamp/blob/main/mmlab2/mmdet/rtmdet_balloon/rtmdet_balloon.ipynb)

##### 生成coco格式标注文件

```shell
python via2coco.py
```

##### 训练

```shell
python tools/train.py rtmdet_tiny_1xb12-40e_balloon.py
```
| config   | checkpoint | log | mAP|
|:--------:|:----------:|:---------:|:---------:|
| rtmdet_tiny_1xb12-40e_balloon.py| best_coco_bbox_mAP_epoch_30.pth| 20230610_052939.log| 70.1|

##### 测试
```shell
python tools/test.py \
    rtmdet_tiny_1xb12-40e_balloon.py \
    work_dirs/rtmdet_tiny_1xb12-40e_balloon/best_coco_bbox_mAP_epoch_30.pth
```

##### 可视化

```shell
python demo/boxam_vis_demo.py \
    resized_image.jpg \
    ../mmdetection/rtmdet_tiny_1xb12-40e_balloon.py \
    ../mmdetection/work_dirs/rtmdet_tiny_1xb12-40e_balloon/best_coco_bbox_mAP_epoch_30.pth \
    --target-layer neck.out_convs[1]
```

![balloon](rtmdet_balloon/balloon.png)

#### 饮料目标检测
##### Colab教程

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liweiwp/OpenMMLabCamp/blob/main/mmlab2/mmdet/rtmdet_drink/rtmdet_drink.ipynb)

##### 训练

```shell
python tools/train.py rtmdet_tiny_1xb12-40e_drink.py
```
| config   | checkpoint | log | mAP|
|:--------:|:----------:|:---------:|:---------:|
| rtmdet_tiny_1xb12-40e_drink.py| best_coco_bbox_mAP_epoch_40.pth| 20230610_084944.log| 94.2|

##### 测试
```shell
python tools/test.py \
    rtmdet_tiny_1xb12-40e_drink.py \
    work_dirs/rtmdet_tiny_1xb12-40e_drink/best_coco_bbox_mAP_epoch_40.pth
```

##### 可视化

```shell
python demo/boxam_vis_demo.py \
    resized_image.jpg \
    ../mmdetection/rtmdet_tiny_1xb12-40e_drink.py \
    ../mmdetection/work_dirs/rtmdet_tiny_1xb12-40e_drink/best_coco_bbox_mAP_epoch_40.pth \
    --target-layer neck.out_convs[1]
```

![drink](rtmdet_drink/drink.png)