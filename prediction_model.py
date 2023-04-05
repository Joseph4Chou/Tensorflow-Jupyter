#!/usr/bin/python

""" 简易的时间序列预测模型

    采用Tensorflow框架, 通过已训练好的model对输入数据开展预测。为了简便起见, 未定义任何类、函数, 编程逻辑也采用平铺直叙方式。

    用法：
    测试集与本文件应放于同一文件夹, 修改fname赋值语句中的文件名称, 预测模型为预先训练好并保存的self_test_conv1d.h5, 主要预测均值
    和标准偏差, 对预测集的前10个数据与实际值绘制折线图对比。
    在使用该模型前, 需安装numpy、tensorflow和matplotlib, 建议使用conda安装并在虚拟环境中运行:
    $python3 prediction_model.py
"""

import os
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# 若替换测试集, 请修改以下赋值语句中的文件名称。
fname = os.path.join("2023.01.03-2.csv")
with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(f"header of csv files: {header}")
print(f"length of lines: {lines[-1]}")

meta_datasets = np.zeros((len(lines),))
print(f"shape of meta_datasets: {meta_datasets.shape}")
raw_data = np.zeros((len(lines), len(header) - 1))
print(f"shape of raw_data: {raw_data.shape}")

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    meta_datasets[i] = values[0]
    raw_data[i, :] = values[:]

#plt.plot(range(len(meta_datasets)), meta_datasets)
#plt.show()
#plt.plot(range(1440), meta_datasets[:1440])
#plt.show()

mean = raw_data[:].mean(axis=0)
print(f"Mean of real data: {mean[0]:.2f}")  
raw_data -= mean
std = raw_data[:].std(axis=0)
print(f"S.D. of real data: {std[0]:.2f}")
raw_data /= std

sampling_rate = 1
sequence_length = 256
delay = sampling_rate * (sequence_length + 24 - 1)
#print(f"delay = {delay}")
batch_size = 128

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=None,
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0)

# 若替换预测模型, 请修改以下赋值语句中的文件名称。
model = keras.models.load_model("self_test_conv1d.h5")
predictions = model.predict(test_dataset)
print(f"Mean of predictions: {predictions.mean(axis=0)}")
print(f"S.D. of predictions: {predictions.std(axis=0)}")

predict_shape = predictions.shape[0]
real_data = meta_datasets[:10]
steps = range(0,10)
plt.figure()
plt.plot(steps, real_data, "r", label="real_data")
plt.plot(steps, predictions[:10], "b", label="predic_data")
plt.title("Predictions")
plt.legend()
plt.show()