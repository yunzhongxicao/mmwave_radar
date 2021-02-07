"""
@File  :auto_correlation.py
@Author:dfc
@Date  :2021/2/115:31
@Desc  :这里主要测试自相关分析在提取信号特征效果
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from read_raw_data import radar_data_read
para_path = '../device_data/2021_1_30_16_11_39_para.txt'
data_path = '../device_data/2021_1_30_16_11_39_data.npy'
# para_path = '../device_data/2021_1_30_11_33_58_para.txt'
# data_path = '../device_data/2021_1_30_11_33_58_data.npy'
para, dem = radar_data_read(para_path, data_path)

num_samples_per_chirp = int(para[0][1])
num_chirps_per_frame = int(para[1][1])
adc_samplerate_hz = int(para[2][1])
frame_period_us = int(para[3][1])
lower_frequency_kHz = int(para[4][1])
upper_frequency_kHz = int(para[5][1])
bgt_tx_power = int(para[6][1])
rx_antenna_mask = int(para[7][1])
chirp_to_chirp_time_100ps = int(para[8][1])
if_gain_dB = int(para[9][1])
frame_end_delay_100ps = int(para[10][1])
shape_end_delay_100ps = int(para[11][1])
num_frames = int(para[12][1])
chirp_time_in_gui = int(para[13][1])

radar_data = dem

# 不同天线的数据
radar_data_1 = radar_data[:, 0:num_samples_per_chirp]
radar_data_2 = radar_data[:, num_samples_per_chirp:2 * num_samples_per_chirp]
radar_data_3 = radar_data[:, num_samples_per_chirp * 2:3 * num_samples_per_chirp]
print("radar_data_1.shape: ", radar_data_1.shape)

# 明确每个chirp和frame的时间
chirp_time_sec = (num_samples_per_chirp + 5) / adc_samplerate_hz
num_chirps = num_chirps_per_frame * num_frames
if chirp_to_chirp_time_100ps == 0:
    time_between_chirp_sec = shape_end_delay_100ps / (10 ** 10) + chirp_time_sec
else:
    time_between_chirp_sec = chirp_to_chirp_time_100ps / (10 ** 10)

if frame_period_us == 0:
    time_between_frame_sec = time_between_chirp_sec * num_chirps_per_frame + frame_end_delay_100ps / (10 ** 10)
else:
    time_between_frame_sec = frame_period_us / (10 ** 6)

print("chirp-time:",time_between_chirp_sec)
print("frame-time:",time_between_frame_sec)
"""
# 时域序列矩阵 
"""
time_domain_matrix = np.zeros([num_chirps, num_samples_per_chirp])
time_resolution_sec = 1 / adc_samplerate_hz
time_domain_row = np.arange(0, num_samples_per_chirp * time_resolution_sec, time_resolution_sec)
for i in range(num_frames):
    for j in range(num_chirps_per_frame):
        time_domain_matrix[i * num_chirps_per_frame + j,
        :] = time_domain_row + j * time_between_chirp_sec + i * time_between_frame_sec
# print("time_domain_matrix:\n", time_domain_matrix)

"""
频域转化为距离
"""
f = np.arange(int(num_samples_per_chirp / 2)) * adc_samplerate_hz / num_samples_per_chirp
c = 3 * (10 ** 8)  # 光速
S = (upper_frequency_kHz - lower_frequency_kHz) * 1000 / chirp_time_sec
d = f * c / 2 / S
d_resolution = adc_samplerate_hz / num_samples_per_chirp * c / 2 / S
print("d_resolution: ", d_resolution)

"""
对每个chirp进行FFT
"""
radar_data_1_fft = np.zeros([num_chirps, num_samples_per_chirp], dtype=complex)
for i in range(num_chirps):
    # standard
    data_tmp = signal.detrend(radar_data_1[i, :])
    data_tmp = data_tmp - data_tmp.mean()
    # fft
    sp = np.fft.fft(data_tmp)
    # 滤波
    for k in range(len(sp)):
        if k >= (len(sp) - 5) or k <= 5:
            sp[k] = 0
    sp = (sp - sp.mean()) / num_samples_per_chirp * 2
    sp[0] = sp[0] / 2
    radar_data_1_fft[i, :] = sp

radar_data_1_fft_amp = np.abs(radar_data_1_fft[:, :int(num_samples_per_chirp / 2)])
radar_data_1_fft_body = radar_data_1_fft[:,21]
radar_data_1_fft_body = (radar_data_1_fft_body- radar_data_1_fft_body.mean())
plt.scatter(radar_data_1_fft_body.real[2 * num_chirps_per_frame:7 * num_chirps_per_frame],
            radar_data_1_fft_body.imag[2 * num_chirps_per_frame:7 * num_chirps_per_frame], s=2)
plt.xlabel("real")
plt.ylabel("imag")
plt.show()

plt.scatter(radar_data_1_fft_body.real[1 * num_chirps_per_frame:20 * num_chirps_per_frame],
            radar_data_1_fft_body.imag[1 * num_chirps_per_frame:20 * num_chirps_per_frame], s=2)
plt.xlabel("real")
plt.ylabel("imag")
plt.show()