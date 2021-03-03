import pandas as pd
import numpy as np


def radar_data_read(para_path, data_path):
    """
    :param para_path:雷达参数输入路径
    :param data_path:雷达数据输入路径
    :return para:返回参数array
    :return dem:返回数据numpy.array
    """
    para_num = 14
    para = pd.read_table(para_path, header=None)
    para = np.array(para)
    para = para[:, 0]
    for i in range(para_num):
        para[i] = para[i].split('=')

    dem = np.load(data_path)
    return para, dem


para_path = '../device_data/2021_1_20_23_42_29_para.txt'
data_path = '../device_data/2021_1_20_23_42_29_data.npy'
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

print(radar_data.shape)
print(radar_data[4, :])

