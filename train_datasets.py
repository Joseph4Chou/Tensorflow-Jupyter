#!/usr/bin/python

""" 简易的时间序列训练模型

    采用Tensorflow框架, 使用一维卷积神经网络对过程自检数据进行训练, 并将训练后的模型保存用于预测。为了简便起见, 未定义任何类、函
    数, 编程逻辑也采用平铺直叙方式。

    用法:
    训练数据集与本文件应放于同一文件夹, 修改fname赋值语句中的文件名称。训练后的模型保存为h5文件。 
    在使用该模型前, 需安装numpy、tensorflow和matplotlib, 建议使用conda安装并在虚拟环境中运行:
    $python3 train_datasets.py
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# 若替换训练集, 请修改以下赋值语句中的文件名称。
fname = os.path.join("self_testing_data_2022_20_no_sifting.csv")
with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
#print(f"Check data set header: {header}")

meta_datasets = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    meta_datasets[i] = values[0]
    raw_data[i, :] = values[:]

# 计算用于训练（60%）、验证（20%）和测试（20%）的样本数
num_train_samples = int(0.6 * len(raw_data))
num_val_samples = int(0.2 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

# 数据规范化: 均值为0, 标准差为1
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

# 创建3个数据集，分别用于训练、验证和测试
sampling_rate = 3
sequence_length = 256
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 128

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=meta_datasets[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=meta_datasets[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=meta_datasets[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

# 计算基准值, MAE: np.mean(np.abs(preds - targets))
def evaluate_naive_method(dataset):
    total_abs_err = 0.
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")

# 一维卷积神经网络训练模型
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 12, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 6, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("saved_best_parameters.keras",
                                   save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=1,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("saved_best_parameters.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

# 训练后的模型保存为h5文件
model.save("conv1d_predict_model.h5")

# 查看训练效果
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()