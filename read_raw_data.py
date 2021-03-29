"""
@File  :read_raw_data.py
@Author:dfc
@Date  :2021/1/2119:33
@Desc  :这里主要做的是将数据读入，然后将大概流程跑通
        涉及到select range bin
        将信号处理流程处理成一个函数，方便多组数据输入
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from signalDetection_v2.get_raw_data import radar_data_read, de, find_peak
import glob, os


def signalProcess(data_path, para_path='E:\\基于毫米波的运动监测\\device_data\\2021_2_27_8_55_32_para.txt', range_low=0.3,
                  range_up=1.75):
    """
    # 毫米波雷达数据在get_raw_data.py中写入
    # 这里不涉及雷达的操作，只是读入先前写入的文件
    """
    # 比较清晰的一组数据
    # para_path = '../device_data/2021_1_30_16_11_39_para.txt'
    # data_path = '../device_data/2021_1_30_16_11_39_data.npy'
    # para_path = 'E:\\基于毫米波的运动监测\\device_data\\2021_1_30_11_33_58_para.txt'
    # data_path = 'E:\\基于毫米波的运动监测\\device_data\\output_single_4_cloth1.npy'
    print(data_path, ":")
    # para_path = 'E:\\基于毫米波的运动监测\\device_data\\2021_2_27_8_55_32_para.txt'
    # data_path = data_name
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
    d_max = c * adc_samplerate_hz / 2 / 2 / S
    print("d_resolution: ", d_resolution)
    print("d_max:", d_max)
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

    """
    slow_time - range 图，幅值
    """
    radar_data_1_fft_amp = np.abs(radar_data_1_fft[:, :int(num_samples_per_chirp / 2)])
    print("radar_data_1_fft_amp.shape:", radar_data_1_fft_amp.shape)
    plt.pcolor(d, time_domain_matrix[:, 0], radar_data_1_fft_amp, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Slow time(s)')
    plt.title('Range-slow time amptitude matrix')
    plt.show()
    # 画出第一个chirp的频谱图
    plt.plot(d, radar_data_1_fft_amp[0, :])
    plt.xlabel("d(m)")
    plt.ylabel("Amtitude")
    plt.title("the first  chirp frequency spectrum")
    plt.show()

    """
    slow_time - range 图，相位
    """
    radar_data_1_fft_phase = np.angle(radar_data_1_fft[:, :int(num_samples_per_chirp / 2)])
    plt.pcolor(d, time_domain_matrix[:, 0], radar_data_1_fft_phase, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Slow time(s)')
    plt.title('Range-slow time phase matrix')
    plt.xlim(range_low, )
    plt.show()

    """
    复平面实部虚部散点图
    """
    plt.scatter(radar_data_1_fft.real[1, :], radar_data_1_fft.imag[1, :], s=1)
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.show()

    """
    phase unwrapping 
    """
    for i in range(radar_data_1_fft_phase.shape[1]):
        for j in range(radar_data_1_fft_phase.shape[0] - 1):
            if radar_data_1_fft_phase[j + 1, i] - radar_data_1_fft_phase[j, i] > np.pi:
                radar_data_1_fft_phase[j + 1, i] = radar_data_1_fft_phase[j + 1, i] - 2 * np.pi
            elif radar_data_1_fft_phase[j + 1, i] - radar_data_1_fft_phase[j, i] < (-1) * np.pi:
                radar_data_1_fft_phase[j + 1, i] = radar_data_1_fft_phase[j + 1, i] + 2 * np.pi
            else:
                pass

    """
    slow_time - range 图，相位，after phase unwrapping
    """
    plt.pcolor(d, time_domain_matrix[:, 0], radar_data_1_fft_phase, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Slow time(s)')
    plt.xlim(range_low, )
    plt.title('Range-slow time phase matrix after phase unwrapping')
    plt.show()

    """
    doppler fft
    """
    radar_data_1_doppler_fft = np.zeros([num_chirps, int(num_samples_per_chirp / 2)], dtype=complex)
    for i in range(int(num_samples_per_chirp / 2)):
        data_tmp = radar_data_1_fft_phase[:, i]
        data_tmp = data_tmp - data_tmp.mean()
        # fft
        sp = np.fft.fft(data_tmp)
        sp = (sp - sp.mean()) * 2 / num_chirps
        sp[0] = sp[0] / 2
        radar_data_1_doppler_fft[:, i] = sp

    """
    frequency-range 图
    """
    radar_data_1_doppler_fft_amp = np.abs(radar_data_1_doppler_fft[0:int(num_chirps / 2), :])
    f_doppler = np.arange(int(num_chirps / 2)) / (num_chirps * time_between_chirp_sec)
    f_doppler_resolution = 1 / (num_chirps * time_between_chirp_sec)
    print("f_doppler_resolution:", f_doppler_resolution)
    plt.pcolor(d, f_doppler, radar_data_1_doppler_fft_amp, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Range(m)')
    plt.ylabel('Frequency(Hz)')
    plt.xlim(0.25, )
    plt.ylim(0, 2)
    plt.title('Range-Vibration map')
    plt.show()

    """
    求呼吸心跳频率
    """
    # 方法一 ： 利用 phase unwrapping后的亮度和方差得到range
    radar_data_1_fft_phase_sum = np.sum(np.power(radar_data_1_fft_phase, 2), axis=0)  # 需要平方才有意义
    radar_data_1_fft_phase_var = np.var(radar_data_1_fft_phase, axis=0)
    # max_index_column_sum = np.argmax(radar_data_1_fft_phase_sum)
    # max_index_column_var = np.argmax(radar_data_1_fft_phase_var)
    index_column_sum = radar_data_1_fft_phase_sum.argsort()
    index_column_var = radar_data_1_fft_phase_var.argsort()
    index_column_sum_check = index_column_sum[-5:]
    index_column_var_check = index_column_var[-5:]  # 按照从小到大的顺序排列，最后一个是最大的
    print("index_column_sum[-5:]", index_column_sum[-5:])
    print("index_column_var[-5:]", index_column_var[-5:])  # 将sum和var分别都是前五的输出出来
    d_body = d[index_column_var[-5:]]  # 这里的d_body只是测试
    print(d_body)

    # 将方差中前5个每个与sum比对，找到匹配相同的
    index_d_body = []
    for i in range(len(index_column_var_check)):
        index_tmp = index_column_var_check[-1 * i - 1]
        for j in range(len(index_column_sum_check)):
            if index_column_sum_check[-1 * j - 1] == index_tmp:
                index_d_body.append(index_tmp)
                break

    if not index_d_body:
        index_d_body.append(index_column_var[-1])

    print("index_d_body:", index_d_body)  # 按照从大到小的顺序排列
    d_body = d[index_d_body]
    print("d_body:", d_body)
    # 将index对应的range取出来
    radar_data_1_doppler_fft_amp_body = radar_data_1_doppler_fft_amp[:, index_d_body[0]]
    radar_data_1_doppler_fft_amp_body = signal.detrend(radar_data_1_doppler_fft_amp_body)
    radar_data_1_doppler_fft_amp_body = radar_data_1_doppler_fft_amp_body - radar_data_1_doppler_fft_amp_body.mean()
    # 作出对应的频谱图
    plt.plot(f_doppler[0:], radar_data_1_doppler_fft_amp_body[0:])
    plt.xlim(0, 4)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("A")
    plt.title("the chosen range {:.2f} m frequency spectrum".format(d_body[0]))
    plt.show()
    # 求出对应范围内的频率
    heartbeat_frequency_1 = find_peak(f_doppler, radar_data_1_doppler_fft_amp_body, 1, 2)
    respire_frequency_1 = find_peak(f_doppler, radar_data_1_doppler_fft_amp_body, 0.1, 0.6)

    print("select range bin 方案一得到的呼吸频率：", round(respire_frequency_1, 2), "---", round(respire_frequency_1 * 60, 2))
    print("select range bin 方案一得到的心跳频率：", round(heartbeat_frequency_1, 2), "---", round(heartbeat_frequency_1 * 60, 2))

    # 方法二：利用doppler FFT后得到的图找出最大平均功率
    radar_data_1_doppler_fft_amp_sum = np.sum(np.power(radar_data_1_doppler_fft_amp, 2), axis=0)
    index_column_doppler_sum = radar_data_1_doppler_fft_amp_sum.argsort()
    index_column_doppler_sum_check = index_column_doppler_sum[-5:]  # 从小到大排列
    index_column_doppler_sum_check = index_column_doppler_sum_check[::-1]  # 从大到小排序
    print("index_column_doppler_sum_check:", index_column_doppler_sum_check)
    d_body_doppler = d[index_column_doppler_sum_check]
    print(d_body_doppler)
    # 取出对应的range
    radar_data_1_doppler_fft_amp_body = radar_data_1_doppler_fft_amp[:, index_column_doppler_sum_check[0]]
    radar_data_1_doppler_fft_amp_body = signal.detrend(radar_data_1_doppler_fft_amp_body)
    radar_data_1_doppler_fft_amp_body = radar_data_1_doppler_fft_amp_body - radar_data_1_doppler_fft_amp_body.mean()
    # 作出对应的频谱图
    plt.plot(f_doppler[5:], radar_data_1_doppler_fft_amp_body[5:])
    plt.xlim(1, 4)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("A")
    plt.title("the chosen range {:.2f} m frequency spectrum by method 2".format(d_body_doppler[0]))
    plt.show()
    # 求出频率
    heartbeat_frequency_2 = find_peak(f_doppler, radar_data_1_doppler_fft_amp_body, 1, 2)
    respire_frequency_2 = find_peak(f_doppler, radar_data_1_doppler_fft_amp_body, 0.1, 0.6)

    print("select range bin 方案二得到的呼吸频率：", round(respire_frequency_2, 2), "---", round(respire_frequency_2 * 60, 2))
    print("select range bin 方案二得到的心跳频率：", round(heartbeat_frequency_2, 2), "---", round(heartbeat_frequency_2 * 60, 2))

    """
    重新画实部虚部散点图
    """
    plt.scatter(
        radar_data_1_fft.real[:, index_column_doppler_sum_check[0]][1 * num_chirps_per_frame:30 * num_chirps_per_frame],
        radar_data_1_fft.imag[:, index_column_doppler_sum_check[0]][1 * num_chirps_per_frame:30 * num_chirps_per_frame],
        s=2)
    plt.xlabel("real")
    plt.ylabel("imag")
    plt.title("the chosen range {:.2f} m real&imag ".format(d_body_doppler[0]))
    plt.show()

    """
    直接对range 位移序列进行自相关分析
    """
    radar_data_1_fft_phase_body = radar_data_1_fft_phase[:, index_column_doppler_sum_check[0]]
    radar_data_1_fft_phase_body_corr = np.correlate(radar_data_1_fft_phase_body, radar_data_1_fft_phase_body,
                                                    mode='full')
    radar_data_1_fft_phase_body_lags = [i * time_between_chirp_sec for i in
                                        range(-len(radar_data_1_fft_phase_body) + 1,
                                              len(radar_data_1_fft_phase_body))]
    print("radar_data_1_fft_phase_body_corr.length: ", len(radar_data_1_fft_phase_body_corr))
    print("radar_data_1_fft_phase_body_lags.length: ", len(radar_data_1_fft_phase_body_lags))
    plt.plot(radar_data_1_fft_phase_body_lags, radar_data_1_fft_phase_body_corr)
    plt.title("the chosen range {:.2f} m auto-correlate  ".format(d_body_doppler[0]))
    plt.show()

    # # 对自相关系数进行FFT
    data_tmp = radar_data_1_fft_phase_body_corr - radar_data_1_fft_phase_body_corr.mean()
    radar_data_1_fft_phase_body_corr_length = len(radar_data_1_fft_phase_body_corr)  # 注意自相关变换后系数长度已经变成 2*num_chirps -1
    sp = np.fft.fft(data_tmp)
    sp = (sp - sp.mean()) * 2 / radar_data_1_fft_phase_body_corr_length
    sp[0] = sp[0] / 2
    radar_data_1_fft_phase_body_corr_fft = sp
    radar_data_1_fft_phase_body_corr_fft_amp = np.abs(radar_data_1_fft_phase_body_corr_fft \
                                                          [0:int(radar_data_1_fft_phase_body_corr_length / 2)])
    f_corr = np.arange(int(radar_data_1_fft_phase_body_corr_length / 2)) * \
             (1 / time_between_chirp_sec) / \
             radar_data_1_fft_phase_body_corr_length  # 这是自相关系数FFT后的频率序列
    plt.plot(f_corr[5:], radar_data_1_fft_phase_body_corr_fft_amp[5:])
    plt.xlim(0, 4)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("A")
    plt.title("the chosen range {:.2f} m frequency spectrum after auto-correlate".format(d_body[0]))
    plt.show()
    # 求频率
    heartbeat_frequency_3 = find_peak(f_corr, radar_data_1_fft_phase_body_corr_fft_amp, 1, 2)
    respire_frequency_3 = find_peak(f_corr, radar_data_1_fft_phase_body_corr_fft_amp, 0.1, 0.6)

    print("直接自相关得到的呼吸频率：", round(respire_frequency_3, 2), "---", round(respire_frequency_3 * 60, 2))
    print("直接自相关得到的心跳频率：", round(heartbeat_frequency_3, 2), "---", round(heartbeat_frequency_3 * 60, 2))

    # DE算法
    heartbeat_frequency_4, heartbeat_frequency_5 = de(time_between_chirp_sec, radar_data_1_fft_phase_body)
    print("基于DE算法的心跳频率：", round(heartbeat_frequency_4, 2), "---", round(heartbeat_frequency_4 * 60, 2))
    print("基于DE算法2的心跳频率：", round(heartbeat_frequency_5, 2), "---", round(heartbeat_frequency_5 * 60, 2))

    print()
    print()
    print()
    print()
    print()


if __name__ == "__main__":
    path = r'E:\基于毫米波的运动监测\device_data\2021_03_01_measure'
    file = glob.glob(os.path.join(path, '*.npy'))
    print(file)
    signalProcess('E:\\基于毫米波的运动监测\\device_data\\2021_03_01_measure\\90cm_140cm_double.npy'
                  , 'E:\\基于毫米波的运动监测\\device_data\\2021_03_01_measure\\para.txt')
