import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
import time
import datetime

# Data load
data_raw = np.loadtxt('E:\YangZT\Robolab\MMW_Radar\InfineonRadar\GUI_Data\BGT60TR13C_record_20200807-111515.raw.txt',
					  dtype=float, comments='#')

# Data split
N_sample = 128
N_chirp = 64
N_frame = int(np.size(data_raw) / 8192)
data = np.resize(data_raw, (N_frame, 64, 128))

# Range FFT
fs = 1000000  # 采样频率1MHz
n = np.arange(0, N_sample, 1)
f = n * fs / N_sample
c = 3 * 10 ** 8  # 电磁波传播速度
S = 1 * 10 ** 9 / (133 * 10 ** (-6))  # 扫频斜率 60-61 GHz
dist = f * c / (2 * S)

data_fft = np.zeros((N_frame, N_chirp, N_sample), dtype=complex)

for i in range(0, N_frame):
	for j in range(0, N_chirp):
		data_temp = data[i, j, :] = data[i, j, :] - np.mean(data[i, j, :])
		data_fft[i, j, :] = fft(data_temp, N_sample)

# np.savetxt('20200805_23_17_data_fft.txt',np.abs(data_fft[0,:,:]))

# Range Doppler
fc = int(1 / (505.64 * 10 ** (-6)))  # Tc=505.64 μs,
n_chirp = np.arange(-N_chirp // 2, N_chirp // 2 + 1, 1)
f_chirp = n_chirp * fc / N_chirp
v_chirp = 2.5 * 10 ** (-3) * f_chirp

data_RD_fft = np.zeros((N_frame, N_chirp, N_sample), dtype=complex)
data_speed = np.zeros((N_frame, 1), dtype=float)
peak_angle = np.zeros((N_frame, 1), dtype=float)
# data_RD_fft_new = np.zeros((N_chirp,N_sample), dtype=complex)

for i in range(0, N_frame):
	for j in range(0, N_sample):
		data_fft_temp = data_fft[i, :, j] - np.mean(data_fft[i, :, j])
		data_RD_fft[i, :, j] = fft(data_fft_temp, N_chirp)

	pos = np.unravel_index(np.argmax(data_RD_fft[i, 0:int(N_chirp / 2), 0:int(N_sample / 2)]),
						   np.shape(data_RD_fft[i, :, :]))  # 查找最大的目标点
	# data_speed[i] = v_chirp[pos[0]]
	peak_angle[i] = np.angle(data_RD_fft[i, pos[0], pos[1]])

	# Range Doppler 正负速度拼接,
	data_RD_fft[i, :, :] = np.r_[data_RD_fft[i, N_chirp // 2:N_chirp, :], data_RD_fft[i, 0:N_chirp // 2, :]]

	# 显示 Range Doppler
	# plt.subplot(2,1,1)
	pcm = plt.pcolormesh(dist[0:int(N_sample / 2)], v_chirp, np.abs(data_RD_fft[i, :, 0:int(N_sample / 2)]),
						 cmap='plasma', vmin=0.0)  # 'PuBu_r'
	plt.colorbar(pcm)
	plt.title("Figure %d " % (i))
	plt.xlabel("Range - m")
	plt.ylabel("Doppler V - m/s")
	'''
    # 显示 Range 峰值对应的相位
    plt.subplot(2,1,2)
    plt.plot(n_chirp, np.angle(data_fft[i, :, pos[1]]))
    plt.xlabel("chirp ID")
    plt.ylabel("Peak phase angle") 
    '''
	plt.draw()
	plt.savefig('E:\YangZT\Robolab\MMW_Radar\InfineonRadar\GUI_Data\ForwFalling5\Figure %d.png' % (i + 1))
	plt.close()

# Speed - time 
n_frame = np.arange(0, N_frame, 1)
n_frame_time = 0.2 * n_frame
plt.title("Falling ")
plt.xlabel("Time - s")
plt.ylabel("Max Speed - m/s")
plt.plot(n_frame_time, data_speed)
# plt.savefig('None person 2020_8_6_15_45.png')
# plt.show()

print(data.shape)
