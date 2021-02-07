import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.font_manager import FontProperties

# 设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

# import data
file_path = '../gui_data/BGT60TR13C_record_20210116-214043.raw.txt'
data = pd.read_csv(file_path,
                   header=None, skiprows=list(range(28)), engine='python')

print(data.head())
data = np.array(data)
data = data[:, 0]
print('data.shape:', data.shape)
print(data[:10])

# import para
para_num = 27
para = pd.read_table(file_path, header=None, nrows=para_num)
# print(para)
para = np.array(para)
para = para[:, 0]
# print(para)
for i in range(para_num):
    para[i] = para[i].split(' = ')

Chirps_per_Frame = int(para[21][1])
Samples_per_Chirp = int(para[22][1])
Samples_per_Frame = int(Chirps_per_Frame * Samples_per_Chirp)
Lower_RF_Frequency_kHz = (float(para[15][1]))
Upper_RF_Frequency_kHz = (float(para[16][1]))
Sampling_Frequency_kHz = int(para[17][1])
Chirp_Time_sec = float(para[24][1])  # chirp time
Pulse_Repetition_Time_sec = float(para[25][1])  # chirp time + pulse time
Frame_Period_sec = float(para[26][1])
Frame_num = 40

# data transform
data_trans = np.zeros((Frame_num, Samples_per_Frame))
print('data_trans.shape:', data_trans.shape)

for i in range(Frame_num):
    data_temp = data[i * (Samples_per_Frame + 1):(i + 1) * (Samples_per_Frame + 1)]  # 每一个frame的数据
    data_temp = data_temp[1:]  # 去除第一行frame
    data_trans[i, :] = data_temp.tolist()  # 每一行是一个frame数据，后续要分每个chirp

# # time domain plot
t = 1 / (Sampling_Frequency_kHz * 1000) * np.arange(Samples_per_Frame)
plt.plot(t[:1600], data_trans[0, :1600])
plt.xlabel("t(s)")
plt.ylabel("data_trans")
plt.show()

# 每个chirp
phase_complex = np.zeros((int(Frame_num * Samples_per_Frame / Samples_per_Chirp), Samples_per_Chirp),
                         dtype=complex)  # 用于存放每个chirpFFT变换后的复数矩阵
# 频率转化为距离
f = np.arange(int(Samples_per_Chirp / 2)) * (Sampling_Frequency_kHz * 1000) / Samples_per_Chirp
c = 3 * (10 ** 8)  # 光速
S = (Upper_RF_Frequency_kHz - Lower_RF_Frequency_kHz) / Chirp_Time_sec
S = S * 1000
d = f * c / 2 / S

# 对每个chirp进行操作
for i in range(Frame_num):
    for j in range(Chirps_per_Frame):
        data_trans_i_j = data_trans[i, j * Samples_per_Chirp:(j + 1) * Samples_per_Chirp]  # 一个chirp数据
        # print(data_trans_i_j.shape)
        # standard
        data_trans_i_j = signal.detrend(data_trans_i_j)
        data_mean = data_trans_i_j.mean()
        data_trans_i_j = data_trans_i_j - data_mean

        # fft
        sp = np.fft.fft(data_trans_i_j)

        # 滤波
        # for k in range(len(sp)):
        # 	if k >=(len(sp) - 5) or k <= 5:
        # 		sp[k] = 0

        # 实部虚部
        sp_real = sp.real
        # print(sp_real.shape)
        sp_real_mean = sp_real.mean()
        sp_real = sp_real - sp_real_mean

        sp_imag = sp.imag
        sp_imag_mean = sp_imag.mean()
        sp_imag = sp_imag - sp_imag_mean

        # 合并实部虚部
        sp = np.vectorize(complex)(sp_real, sp_imag)
        # print(sp.shape)  # (256,)
        phase_complex[i * Chirps_per_Frame + j, :] = sp.tolist()

        P2 = np.abs(sp / Samples_per_Chirp) * 2
        P1 = P2[range(int(Samples_per_Chirp / 2))]

        # plot
        if i == 0 and j <= 1:
            plt.plot(d, P1)
            plt.xlabel("d(m)")
            plt.ylabel("P1")
            plt.xlim(0, 1.5)
            plt.show()

# range-slow time matrix
phase_angle_temp = np.angle(phase_complex, deg=False)
phase_angle = phase_angle_temp[:, :int(Samples_per_Chirp / 2)]  # 对称，取一半
print("phase_angle.shape:", phase_angle.shape)
phase_real = phase_complex[:, :int(Samples_per_Chirp / 2)].real  # 实部，且取一半
phase_imag = phase_complex[:, :int(Samples_per_Chirp / 2)].imag
phase_abs = np.abs(phase_complex[:, :int(Samples_per_Chirp / 2)])
print(phase_abs.shape)  # (5120,128)	5120个chirp，每个chirp128个samples

plt.scatter(phase_real[:64, :], phase_imag[:64, :], s=1)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("real")
plt.ylabel("imag")
plt.show()

T_d_sample = Pulse_Repetition_Time_sec  # 位移采样周期
f_d_sample = 1 / T_d_sample  # 位移采样频率
time_d = T_d_sample * np.arange(int(Frame_num * Samples_per_Frame / Samples_per_Chirp))  # 离散的纵向位移时间序列

plt.pcolor(d, time_d, phase_abs, cmap=plt.cm.jet)  # 以绝对值为区分度
plt.colorbar()
plt.xlabel('Range(m)')
plt.ylabel('Slow time(s)')
plt.xlim(0,1.5)
plt.title('Range-slow time matrix')
plt.show()

plt.pcolor(d, time_d, phase_angle, cmap=plt.cm.jet)  # 以相位为区分度
plt.colorbar()
plt.xlabel('Range(m)')
plt.ylabel('Slow time(s)')
plt.title('Range-slow time phase matrix')
plt.show()

# phase unwrapping
phase_angle_unwrap = np.zeros(phase_angle.shape)   # 存储phase unwarpping后的相位
for i in range(phase_angle.shape[1]):

    for j in range(phase_angle.shape[0] - 1):

        if phase_angle[j + 1, i] - phase_angle[j, i] > np.pi:
            phase_angle[j + 1, i] = phase_angle[j + 1, i] - 2 * np.pi

        elif phase_angle[j + 1, i] - phase_angle[j + 1, i] < (-1) * np.pi:
            phase_angle[j + 1, i] = phase_angle[j + 1, i] + 2 * np.pi

        else:
            pass

plt.pcolor(d, time_d, phase_angle, cmap=plt.cm.jet)  # 以相位为区分度
plt.colorbar()
plt.xlabel('Range(m)')
plt.ylabel('Slow time(s)')
plt.title('unwrapping phase')
plt.show()

# doppler fft
f_doppler = np.arange(int(phase_angle.shape[0] / 2)) * (f_d_sample / phase_angle.shape[0])
phase_angle_fft_abs = np.zeros((int(phase_angle.shape[0] / 2), phase_angle.shape[1]))
for j in range(phase_angle.shape[1]):
    phase_angle_fft_j = np.fft.fft(phase_angle[:, j])
    phase_angle_fft_j_P2 = np.abs(phase_angle_fft_j / phase_angle.shape[0]) * 2
    phase_angle_fft_j_P1 = phase_angle_fft_j_P2[:int(phase_angle.shape[0] / 2)]
    phase_angle_fft_abs[:, j] = phase_angle_fft_j_P1.tolist()

plt.pcolor(d, f_doppler, phase_angle_fft_abs, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel('Range(m)')
plt.ylabel('Frequency')
plt.ylim(0, 3)
plt.xlim(0.2, 0.7)
plt.title('Range-Vibration map')
plt.show()

max_index = []  # 矩阵最大值的索引
phase_angle_fft_abs_sum = np.sum(phase_angle_fft_abs, axis=0)

# max_index.append(np.where(phase_angle_fft_abs == np.max(phase_angle_fft_abs))[0])
# max_index.append(np.where(phase_angle_fft_abs == np.max(phase_angle_fft_abs))[1])
# max_index_column = max_index[1][0]
max_index_column = np.argmax(phase_angle_fft_abs[1,:])

print('max_column_index:', max_index_column)

d_body = d[max_index_column]  # 身体胸腔对应的距离
print('d_body:', d_body)

phase_angle_fft_abs_body = phase_angle_fft_abs[:, max_index_column]
phase_angle_fft_abs_body = signal.detrend(phase_angle_fft_abs_body)
phase_angle_fft_abs_body_mean = phase_angle_fft_abs_body.mean()
phase_angle_fft_abs_body = phase_angle_fft_abs_body - phase_angle_fft_abs_body_mean
plt.plot(f_doppler[1:], phase_angle_fft_abs_body[1:])
plt.xlim(0, 5)
plt.ylabel('A')
plt.xlabel('Frequency / Hz')
plt.show()

print('心跳频率：', f_doppler[3])
print('呼吸频率：', f_doppler[1])
