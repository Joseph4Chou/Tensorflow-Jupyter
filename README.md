# Tensorflow-Jupyter

使用自检数据训练一维卷积模型，预测未来产品的质量。

目前进度：

在 train_datasets.py 中使用 self_test_2022.csv 对一维卷积模型进行了训练，训练次数较多，预测结果质量稳定性超出预期和实际。目前，将训练次数减少到1次，训练后的模型保存为 self_test_conv1d.h5 。

在 prediction_model.py 中使用 self_test_conv1d.h5 模型进行预测，预测数据为1个班次的自检数据，预测结果与实际值较接近。

| 组别 | 预测均值 | 预测SD | 实际均值 | 实际SD |
|:----|:--------:|:------:|:-------:|------:|
| 1 | 4200.12 | 92.45 | 4206.40 | 118.36 |
| 2 | 4228.32 | 95.06 | 4222.71 | 120.66 |
| 3 | 4252.09 | 30.39 | 4209.74 | 110.74 |
| 4 | 4295.23 | 59.53 | 4208.07 | 116.88 |
| 5 | 4255.72 | 57.17 | 4209.51 | 120.84 |
| 6 | 4285.19 | 49.65 | 4209.11 | 124.62 |