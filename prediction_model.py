#!/usr/bin/python
"""
"""

import os
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# import meta data from csv files
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

# enumerate()枚举序列中的内容, 返回下标i和内容line
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