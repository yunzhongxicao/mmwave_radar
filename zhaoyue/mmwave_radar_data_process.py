# import radar_data_write
from ctypes import *
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.fftpack import fft, ifft
from scipy import signal
import seaborn as sns
import pandas as pd


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


if __name__ == "__main__":

    # =====================================================================================
    # 毫米波雷达数据写入在get_raw_data.py中完成
    # 这里只负责将cvs、txt文件读入，不涉及雷达连接的操作
    # =====================================================================================
    para_path = '../device_data/2021_1_16_10_14_para.txt'
    data_path = '../device_data/2021_1_16_10_14_data.npy'
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

    # =====================================================================================
    # 读取数据文件，包含3个天线数据
    # =====================================================================================
    radar_data = dem

    # num_samples_per_chirp = 128
    # num_chirps_per_frame = 512
    # adc_samplerate_hz = 2000000
    # frame_period_us = 0
    # lower_frequency_kHz = 58000000
    # upper_frequency_kHz = 63000000
    # bgt_tx_power = 31
    # rx_antenna_mask = 7
    # chirp_to_chirp_time_100ps = 0
    # if_gain_dB = 33
    # frame_end_delay_100ps = 1500000
    # shape_end_delay_100ps = 1500000
    # num_frames = 100

    radar_data_1 = radar_data[:, 0: num_samples_per_chirp]
    # radar_data_2 = radar_data[:, num_samples_per_chirp  : (2 * num_samples_per_chirp)]
    # radar_data_3 = radar_data[:, (2 * num_samples_per_chirp) : (3 * num_samples_per_chirp)]
    # print(radar_data_1.shape)
    # print(radar_data_2.shape)
    # print(radar_data_3.shape)

    # =====================================================================================
    # 测量参数
    # =====================================================================================
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

    # =====================================================================================
    # 时域序列矩阵：矩阵中的每个值对应测量的数据矩阵响应位置的时间点，单位为s
    # =====================================================================================
    time_domain_matrix = np.zeros([num_chirps, num_samples_per_chirp])
    time_resolution_sec = 1 / adc_samplerate_hz
    time_domain_row = np.arange(0, num_samples_per_chirp * time_resolution_sec, time_resolution_sec)
    for i in range(num_frames):
        for j in range(num_chirps_per_frame):
            time_domain_matrix[i * num_chirps_per_frame + j,
            :] = i * time_between_frame_sec + j * time_between_chirp_sec + time_domain_row
    # print(time_domain_matrix)

    # =====================================================================================
    # 画时域图
    # =====================================================================================
    # radar_data_list = radar_data_1.flatten()
    # fasttime_list = time_domain_matrix.flatten()
    # plt.figure(num = 1)
    # plt.plot(fasttime_list, radar_data_list)
    # plt.xlabel("t(s)")
    # plt.ylabel("radar_data_1")
    # plt.show()

    # =====================================================================================
    # fast time fft
    # =====================================================================================    
    radar_data_1_fft = np.zeros([num_chirps, num_samples_per_chirp], dtype=complex)
    for i in range(num_chirps):
        data_tmp = signal.detrend(radar_data_1[i, :])
        data_tmp = data_tmp - data_tmp.mean()
        data_tmp_2 = fft(data_tmp)
        data_tmp_2 = (data_tmp_2 - data_tmp_2.mean()) * 2 / num_samples_per_chirp
        data_tmp_2[0] = data_tmp_2[0] / 2
        radar_data_1_fft[i, :] = data_tmp_2

    # =====================================================================================
    # 频率转距离
    # =====================================================================================  
    f = np.arange(int(num_samples_per_chirp / 2)) * adc_samplerate_hz / num_samples_per_chirp
    c = 3 * (10 ** 8)  # 光速
    S = (upper_frequency_kHz - lower_frequency_kHz) * 1000 / chirp_time_sec
    d = f * c / 2 / S

    # =====================================================================================
    # slow time-range图，幅值
    # =====================================================================================  
    # radar_data_1_fft_amp = np.abs(radar_data_1_fft[:, 0 : int(num_samples_per_chirp / 2)])
    # plt.figure(num = 2)
    # plt.pcolor(d, time_domain_matrix[:,0], radar_data_1_fft_amp, shading='auto', cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.xlabel('Range(m)')
    # plt.ylabel('Slow time(s)')
    # plt.title('Range-slow time amptitude matrix')
    # plt.show()

    # =====================================================================================
    # slow time-range图，相位
    # =====================================================================================  
    radar_data_1_fft_phase = np.angle(radar_data_1_fft[:, 0: int(num_samples_per_chirp / 2)])
    plt.figure(num=3)
    plt.pcolor(d, time_domain_matrix[:, 0], radar_data_1_fft_phase, shading='auto', cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Slow time(s)')
    plt.title('Range-slow time phase matrix')
    # plt.show()

    # =====================================================================================
    # 复平面散点图
    # =====================================================================================  
    # plt.figure(num = 4)
    # plt.scatter(radar_data_1_fft.real[1000:5000, :], radar_data_1_fft.imag[1000:5000, :], s=1)
    # plt.xlabel('real')
    # plt.ylabel('imag')
    # plt.show()

    # =====================================================================================
    # phase unwrapping
    # =====================================================================================  
    for m in range(int(num_samples_per_chirp / 2)):
        for n in range(1, num_chirps):
            if radar_data_1_fft_phase[n, m] - radar_data_1_fft_phase[n - 1, m] > np.pi:
                radar_data_1_fft_phase[n, m] = radar_data_1_fft_phase[n, m] - 2 * np.pi
            elif radar_data_1_fft_phase[n, m] - radar_data_1_fft_phase[n - 1, m] < -np.pi:
                radar_data_1_fft_phase[n, m] = radar_data_1_fft_phase[n, m] + 2 * np.pi
            else:
                pass

    # =====================================================================================
    # slow time-range图，相位，after phase unwrapping
    # =====================================================================================  
    plt.figure(num=5)
    plt.pcolor(d, time_domain_matrix[:, 0], radar_data_1_fft_phase, shading='auto', cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Slow time(s)')
    plt.title('Range-slow time phase matrix after phase unwrapping')
    # plt.show()

    # =====================================================================================
    # doppler fft
    # =====================================================================================
    radar_data_1_doppler_fft = np.zeros([num_chirps, int(num_samples_per_chirp / 2)], dtype=complex)
    for i in range(int(num_samples_per_chirp / 2)):
        data_tmp = radar_data_1_fft_phase[:, i]
        data_tmp = data_tmp - data_tmp.mean()
        data_tmp_2 = fft(data_tmp)
        data_tmp_2 = (data_tmp_2 - data_tmp_2.mean()) * 2 / num_chirps
        data_tmp_2[0] = data_tmp_2[0] / 2
        radar_data_1_doppler_fft[:, i] = np.abs(data_tmp_2)

    # =====================================================================================
    # frequency-range图
    # =====================================================================================
    radar_data_1_doppler_fft_amp = np.abs(radar_data_1_doppler_fft[0: int(num_chirps / 2), :])
    f_doppler = np.arange(int(num_chirps / 2)) / (num_chirps * time_between_chirp_sec)
    plt.figure(num=6)
    plt.pcolor(d, f_doppler, radar_data_1_doppler_fft_amp, shading='auto', cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Frequency(Hz)')
    plt.xlim(0.5, 0.8)
    plt.ylim(0, 2)
    plt.title('Range-Vibration map')
    # plt.show()

    # =====================================================================================
    # 求心跳呼吸频率
    # =====================================================================================
    phase_angle_fft_sum = np.sum(radar_data_1_fft_phase, axis=0)
    phase_angle_fft_var = np.var(radar_data_1_fft_phase, axis=0)
    max_index_column_sum = np.argmax(phase_angle_fft_sum[1:]) + 1
    max_index_column_var = np.argmax(phase_angle_fft_var[1:]) + 1
    print('max_column_index_sum:', max_index_column_sum)
    print('max_column_index_var:', max_index_column_var)

    phase_angle_fft_amp_sum = np.sum(radar_data_1_doppler_fft_amp[:, 1:], axis=1)
    max_index_row_sum_1 = np.argmax(phase_angle_fft_amp_sum[1:]) + 1
    max_index_row_sum_2 = np.argmax(phase_angle_fft_amp_sum[12:]) + 12
    print('max_row_index_sum_1:', max_index_row_sum_1)
    print('max_row_index_sum_2:', max_index_row_sum_2)
    print('心跳频率：', f_doppler[max_index_row_sum_2])
    print('呼吸频率：', f_doppler[max_index_row_sum_1])

    max_index_column = 20
    d_body = d[max_index_column]  # 身体胸腔对应的距离
    print('d_body:', d_body)

    phase_angle_fft_abs_body = radar_data_1_doppler_fft_amp[:, max_index_column]
    phase_angle_fft_abs_body = signal.detrend(phase_angle_fft_abs_body)
    phase_angle_fft_abs_body_mean = phase_angle_fft_abs_body.mean()
    phase_angle_fft_abs_body = phase_angle_fft_abs_body - phase_angle_fft_abs_body_mean

    max_index_frewuency_1 = np.argmax(phase_angle_fft_abs_body[1:]) + 1
    max_index_frewuency_2 = np.argmax(phase_angle_fft_abs_body[12:]) + 12
    print('心跳频率：', f_doppler[max_index_frewuency_2])
    print('呼吸频率：', f_doppler[max_index_frewuency_1])

    plt.figure(num=7)
    plt.plot(f_doppler[1:], phase_angle_fft_abs_body[1:])
    plt.xlim(0, 5)
    plt.ylabel('A')
    plt.xlabel('Frequency / Hz')
    plt.show()
