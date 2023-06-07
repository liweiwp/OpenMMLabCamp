#### 水果分类
##### 数据集划分

```bash
python split_data.py fruite30_train fruit30
```

##### 微调训练

```bash
python tools/train.py resnet50_finetune.py
```
| config   | checkpoint | log | accuracy/top1|
|:--------:|:----------:|:---------:|:---------:|
| resnet50_finetune.py| epoch_10.txt| 20230607_225305/20230607_225305.log| 84.2342|

##### 推理
```bash
python demo/image_demo.py \
    ../data/fruit30/val/西瓜/100.jpg \
    configs/fruit30/resnet50_finetune.py \
    --checkpoint work_dirs/resnet50_finetune/epoch_10.pth
```
