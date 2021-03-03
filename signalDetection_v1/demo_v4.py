import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.font_manager import FontProperties

# 设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
# import data
file_path = '../gui_data/BGT60TR13C_record_20200625-135019.raw.txt'
data = pd.read_csv(file_path,
				   header=None, skiprows=list(range(28)), engine='python')

print(data.head())
data = np.array(data)
data = data[:, 0]
print(data.shape)
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
Lower_RF_Frequency_kHz = int(para[15][1])
Upper_RF_Frequency_kHz = int(para[16][1])
Sampling_Frequency_kHz = int(para[17][1])
Chirp_Time_sec = float(para[24][1])  # chirp time
Pulse_Repetition_Time_sec = float(para[25][1])  # chirp time + pulse time
Frame_Period_sec = float(para[26][1])
Frame_num = 85

# data transform
data_trans = np.zeros((Frame_num, Samples_per_Frame))
print(data_trans.shape)
# i = 0
# while True:
# 	try:
# 		data_temp = data[i*(Samples_per_Frame+1):(i+1)*(Samples_per_Frame+1)]
# 		print(data_temp)
# 		data_temp = data_temp[1:]
# 		print(data_temp.type)
# 		# data_trans[i] = data_temp
# 		data_trans = data_trans.append(data_temp)
# 		i = i + 1
# 		print(i)
#
# 	except:
# 		break

for i in range(Frame_num):
	data_temp = data[i * (Samples_per_Frame + 1):(i + 1) * (Samples_per_Frame + 1)]
	# print(data_temp)
	data_temp = data_temp[1:]
	if i <= 2:
		print(data_temp.shape)
	# print(type(data_temp))

	# data_trans[i] = data_temp
	# data_trans[i, :] = data_temp.reshape(Samples_per_Frame,1)
	# data_trans[i, :] = data_temp
	data_trans[i, :] = data_temp.tolist()
# print(data_trans.shape)
# print(type(data_trans))
# print(data_trans[0, 0:5])
# data_array = np.array(data_trans)
# print(data_array)


# # time domain plot
t = 1 / (Sampling_Frequency_kHz * 1000) * np.arange(Samples_per_Frame)
plt.plot(t[:1600], data_trans[0, :1600])
plt.title("雷达得到的时域波形",fontproperties=font)
plt.xlabel('时间/s',fontproperties=font)
plt.ylabel('幅值',fontproperties=font)
plt.show()

# 每个chirp
d_all = []
phase_all = []
for i in range(Frame_num):

	for j in range(Chirps_per_Frame):
		data_trans_i_j = data_trans[i, j * Samples_per_Chirp:(j + 1) * Samples_per_Chirp]
		# print(data_trans_i_j.shape)

		# standard
		data_trans_i_j = signal.detrend(data_trans_i_j)
		data_mean = data_trans_i_j.mean()
		data_trans_i_j = data_trans_i_j - data_mean

		#  fft
		sp = np.fft.fft(data_trans_i_j)
		f = np.arange(int(Samples_per_Chirp / 2)) * (Sampling_Frequency_kHz * 1000) / Samples_per_Chirp

		# 滤波
		sp_trans = sp  # 滤波后频域信号
		for k in range(len(sp)):
			if k >= (len(sp) - 5) or k <= 5:
				sp_trans[k] = 0

		P2 = np.abs(sp_trans / Samples_per_Chirp) * 2  # 滤波后频域信号幅值
		P1 = P2[range(int(Samples_per_Chirp / 2))]
		test = np.fft.ifft(sp_trans)  # 滤波后反变换时域信号

		# plot
		# plt.plot(f,P1)
		# plt.show()
		# break

		c = 3 * (10 ** 8)  # 光速
		S = (Upper_RF_Frequency_kHz - Lower_RF_Frequency_kHz) / (Chirp_Time_sec)
		S = S * 1000
		d = f * c / 2 / S
		# plt.plot(d, P1)
		# plt.show()
		# break

		# 找出最大值索引
		max_index = np.argmax(P1)

		d_max = d[max_index]
		d_all.append(d_max)
		# if i <= 5 and j <= 2:
		# 	print('d_max:', d_max)
		# print('d_all:',d_all)
		# 相位
		phase_i_j = np.angle(sp_trans[max_index], deg=False)
		phase_all.append(phase_i_j)

T_d_sample = Pulse_Repetition_Time_sec
f_d_sample = 1 / T_d_sample

# 利用频谱测出的距离
plt.plot(T_d_sample * np.arange(len(d_all)), d_all)
plt.title("利用频谱测出的胸腔位移变化",fontproperties=font)
plt.xlabel('时间/s',fontproperties=font)
plt.ylabel('位移',fontproperties=font)
plt.show()

# phase standard
phase_all = signal.detrend(phase_all)
phase_mean = phase_all.mean()
phase_all = phase_all - phase_mean

# 利用相位测出距离
lamda = 0.005
d_phase = phase_all * lamda / 4 / np.pi
# d_phase = [i *lamda/4/3.14 for  i in phase_all]
# plt.plot(T_d_sample * np.arange(len(d_all)), d_phase)
plt.plot(T_d_sample * np.arange(len(d_all)), phase_all)
plt.title("利用相位测出的胸腔位移变化",fontproperties=font)
plt.xlabel('时间/s',fontproperties=font)
plt.ylabel('位移',fontproperties=font)
plt.show()

# fft得到心跳呼吸频率
phase_all_fft = np.fft.fft(phase_all)
f_phase = np.arange(int(len(phase_all)) / 2) * (f_d_sample / len(phase_all))
phase_all_P2 = np.abs(phase_all_fft / len(phase_all)) * 2
phase_all_P1 = phase_all_P2[range(int(len(phase_all) / 2))]
plt.plot(f_phase[:40], phase_all_P1[:40])
plt.xlabel('f/Hz')
plt.ylabel('d')
plt.show()

# fft得到心跳呼吸频率具体值
sort_phase_index = phase_all_P1.argsort()
for i in range(2):
	if i == 0:
		print('心跳频率', f_phase[sort_phase_index[-1 - i]])

	if i == 1:
		print('呼吸频率', f_phase[sort_phase_index[-1 - i]])
