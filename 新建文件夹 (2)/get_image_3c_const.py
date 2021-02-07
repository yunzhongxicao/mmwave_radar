from ctypes import *
import numpy as np
import torch
import datetime
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import axes3d
import time
import imageio
from PIL import Image
from multiprocessing import Process, Manager, Lock
import os
import math
pi = math.pi

# 定义全局变量
shared_raw_list = []
shared_radar_running_flag = []
img_matrix = []

total_frames = 0
frames_already_scaned = 0
switch_on = []
switch_off = []


# =====================================================================================
## DeviceConfig Structure
# This structure contains all relevant parameters for acquisition of time domain data.
# When a connection to sensor device is established, the device is configured according to the
# parameters of this struct. Each member of this structure is described below.
# -------------------------------------------------------------------------------------
# 'num_samples_per_chirp'         This is the number of samples acquired during each
#                                 chirp of a frame. The duration of a single
#                                 chirp depends on the number of samples and the
#                                 sampling rate.
# -------------------------------------------------------------------------------------                                
# 'num_chirps_per_frame'          This is the number of chirps a single data frame
#                                 consists of. 
# -------------------------------------------------------------------------------------                                
# 'adc_samplerate_hz'             This is the sampling rate of the ADC used to acquire
#                                 the samples during a chirp. The duration of a single
#                                 chirp depends on the number of samples and the
#                                 sampling rate.
# -------------------------------------------------------------------------------------
# 'frame_period_us'               This is the time period that elapses between the
#                                 beginnings of two consecutive frames. The reciprocal
#                                 of this parameter is the frame rate. 
# -------------------------------------------------------------------------------------
# 'lower_frequency_kHz'           This is the start frequency of the FMCW frequency
#                                 ramp.
# -------------------------------------------------------------------------------------
# 'upper_frequency_kHz'           This is the end frequency of the FMCW frequency
#                                 ramp.
# -------------------------------------------------------------------------------------
# 'bgt_tx_power'                  This value controls the power of the transmitted RX
#                                 signal. This is an abstract value between 0 and 31
#                                 without any physical meaning. Refer to BGT60TR13AIP
#                                 data sheet do learn more about the TX power
#                                 BGT60TR13AIP is capable of. 
# -------------------------------------------------------------------------------------
# 'rx_antenna_mask'               In this mask each bit represents one RX antenna of
#                                 BGT60TR13AIP. If a bit is set the according RX
#                                 antenna is enabled during the chirps and the signal
#                                 received through that antenna is captured. 
# -------------------------------------------------------------------------------------
# 'chirp_to_chirp_time_100ps'     This is the time period that elapses between the
#                                 beginnings of two consecutive chirps in a frame.
# -------------------------------------------------------------------------------------
# 'if_gain_dB'                    This is the amplification factor that is applied to
#                                 the IF signal coming from the RF mixer before it is
#                                 fed into the ADC. 
# -------------------------------------------------------------------------------------
# 'frame_end_delay_100ps'         This parameter defines the delay after each frame in
#                                 100 picosecond steps. In order to set this value
#                                 frame_period_us must be set to 0, otherwise this value
#                                 will be ignored.
# -------------------------------------------------------------------------------------
# 'shape_end_delay_100ps'         This parameter defines the delay after each shape in
#                                 100 picosecond steps. In order to set this value
#                                 chirp_to_chirp_time_100ps must be set to 0, otherwise
#                                 this value will be ignored. 
# =====================================================================================
class DeviceConfig(Structure):
    _fields_ = [('num_samples_per_chirp', c_uint32),
                ('num_chirps_per_frame', c_uint32),
                ('adc_samplerate_hz', c_uint32),
                ('frame_period_us', c_uint64),
                ('lower_frequency_kHz', c_uint32),
                ('upper_frequency_kHz', c_uint32),
                ('bgt_tx_power', c_uint8),
                ('rx_antenna_mask', c_uint8),
                ('chirp_to_chirp_time_100ps', c_uint64),
                ('if_gain_dB', c_uint8),
                ('frame_end_delay_100ps', c_uint64),
                ('shape_end_delay_100ps', c_uint64)]


# =====================================================================================
## Matrix Structure
# Defines the structure for one dimensional floating point data array used within API.
# Data length is fixed in this matrix i.e. matrix neither grows nor shrinks.
# The data is arranged sequentially in a row-major order, i.e., all elements of a given row are
# placed in successive memory locations
# -------------------------------------------------------------------------------------
# 'rows'              Number of rows in the matrix
# -------------------------------------------------------------------------------------
# 'columns'           Number of columns in the matrix
# -------------------------------------------------------------------------------------
# 'data'              Pointer to floating point memory containing data values
# =====================================================================================
class Matrix(Structure):
    _fields_ = [('rows', c_uint32),
                ('columns', c_uint32),
                ('data', POINTER(c_float))]

    # returns the value of the matrix element at the specified row and column
    # @param self      The pointer to the Matrix
    # @param row       Row number of required matrix element
    # @param column    Column number of required matrix element
    def at(self, row, column):
        return self.data[row * self.columns + column]


# =====================================================================================
## Frame Structure
# This structure holds a complete frame of time domain data.
# When time domain data is acquired by a radar sensor device, it is copied into an instance of
# this structure. The structure contains one matrix for each enabled RX antenna.
# Each member of this structure is described below.
# -------------------------------------------------------------------------------------
# 'num_rx'              The number of rx matrices in this instance (same as the number
#                       of RX antennas enabled in the radar device)
# -------------------------------------------------------------------------------------
# 'rx_data'             This is an array of data matrices. It contains num_rx elements.
# =====================================================================================
class Frame(Structure):
    _fields_ = [('num_rx', c_uint8),
                ('rx_data', POINTER(Matrix))]

    ## all_antenna Method
    # Returns the pointer to the beginning of the data of the entire frame
    # @param self      The pointer to the Frame
    def all_antenna(self):
        return self.rx_data

    ## per_antenna Method
    # Returns the pointer to the beginning of the data of the the specified antenna
    # @param self      The pointer to the Frame
    # @param antenna   Number of the antenna whose radar is required
    def per_antenna(self, antenna):
        return self.rx_data[antenna]


# =====================================================================================
## mkDevConf Method
# This method returns a DeviceConfig Structure, with its fields having0 the values 
# specified, or if not specified, having default values.
# =====================================================================================
def mkDevConf(num_samples_per_chirp=128,
              num_chirps_per_frame=64,
              adc_samplerate_hz=1000000,
              # 那么，一个chirp的时常为num_samples_per_chirp/adc_samplerate_hz ~ 0.000133s
              frame_period_us=150000,
              lower_frequency_kHz=59700000,
              upper_frequency_kHz=61700000,
              # 这里，上下相减就是扫频的带宽
              bgt_tx_power=31,
              rx_antenna_mask=7,  # 目前是111，则3个天线都打开（测试过不打开，所以应该是可以控制的）
              chirp_to_chirp_time_100ps=5056400,
              # 这个就是chirp时间加上中间的间隔时间，也就是gui中的pulse repetition。间隔时间不可直接设置，间接来自于这个设置
              # 32×1870000×100 + 400000000×100 = ps/10(12)=0.045984s
              if_gain_dB=33,
              frame_end_delay_100ps=400000000,  # 0.04s，每个frame之间的间隔
              shape_end_delay_100ps=1500000):  # 这个值明显是被覆盖了的，而且以上两个值在frame层面，对信号采集已经没有品质上的影响了
    return DeviceConfig(num_samples_per_chirp,
                        num_chirps_per_frame,
                        adc_samplerate_hz,
                        frame_period_us,
                        lower_frequency_kHz,
                        upper_frequency_kHz,
                        bgt_tx_power,
                        rx_antenna_mask,
                        chirp_to_chirp_time_100ps,
                        if_gain_dB,
                        frame_end_delay_100ps,
                        shape_end_delay_100ps)


DeviceHandle = c_void_p

# =====================================================================================
# Error code definitions
# =====================================================================================
IFX_OK = 0  # No error
IFX_ERROR_TIMEOUT = 65559  # Timeout occurred on communication interface with device
IFX_ERROR_FIFO_OVERFLOW = 65560  # FIFO Overflow occurred, signifying potential loss of data 

# =====================================================================================
# Radar SDK API wrappers
# =====================================================================================

# radar_sdk = CDLL(r'C:\Users\bitwa\PycharmProjects\mmradar\video-classification\CRNN\radar_sdk_dll.dll')
radar_sdk = CDLL(r'E:\mmradar\radar_sdk_dll.dll')
radar_sdk.ifx_device_create.restype = c_int
radar_sdk.ifx_device_create.argtypes = [c_void_p, c_void_p]

radar_sdk.ifx_device_destroy.restype = c_int
radar_sdk.ifx_device_destroy.argtypes = [c_void_p]

radar_sdk.ifx_device_create_frame_from_device_handle.restype = c_int
radar_sdk.ifx_device_create_frame_from_device_handle.argtypes = [c_void_p, c_void_p]

radar_sdk.ifx_device_get_next_frame.restype = c_int
radar_sdk.ifx_device_get_next_frame.argtypes = [c_void_p, c_void_p]


# =====================================================================================
## check_rc Method
# Checks and reports if the return code is not 0
# =====================================================================================
def check_rc(rc):
    if rc != IFX_OK:
        raise RuntimeError("Got radar sdk error code {}".format(rc))


# =====================================================================================
## ifx_device_create Method
# Wrapper for ifx_device_create method from radar_sdk
# @param device_config     The DeviceConfig structure with the required device config
#                          values.
# @return handle           A device handle representing the connected device
# =====================================================================================
def ifx_device_create(device_config):
    handle = DeviceHandle()
    check_rc(radar_sdk.ifx_device_create(byref(device_config), byref(handle)))
    return handle


# =====================================================================================
## ifx_device_destroy Method
# Wrapper for ifx_device_destroy method from radar_sdk
# @param handle           A device handle representing the connected device
# =====================================================================================   
def ifx_device_destroy(handle):
    check_rc(radar_sdk.ifx_device_destroy(handle))


# =====================================================================================
## ifx_device_create_frame_from_device_handle Method
# Wrapper for ifx_device_create_frame_from_device_handle method from radar_sdk
# @param    handle            A device handle representing the connected device
# @return   frame             A Frame structure with the format of the data expected
#                             from the connected device specified by the handle
# =====================================================================================
def ifx_device_create_frame_from_device_handle(device_handle):
    frame = Frame()
    check_rc(radar_sdk.ifx_device_create_frame_from_device_handle(device_handle, byref(frame)))
    return frame


# =====================================================================================
## ifx_device_get_next_frame Method
# Wrapper for ifx_device_get_next_frame method from radar_sdk
# @param    handle            A device handle representing the connected device
# @param    frame             A Frame structure with the format of the data expected
#                             from the connected device specified by the handle
# @return   ret               An integer value that represents an error code which
#                             signifies success / failure of the call to get next frame            
# =====================================================================================
def ifx_device_get_next_frame(device_handle, frame):
    ret = radar_sdk.ifx_device_get_next_frame(device_handle, byref(frame))
    if (ret == IFX_ERROR_FIFO_OVERFLOW) or (ret == IFX_ERROR_FIFO_OVERFLOW):
        return ret
    else:
        check_rc(ret)
        return ret


# =====================================================================================
# Example Usage
# =====================================================================================

def compose_gif(gif_path, imgs):
    # img_paths = ["img/1.jpg","img/2.jpg","img/3.jpg","img/4.jpg"
    # ,"img/5.jpg","img/6.jpg"]
    gif_images = []
    for path in imgs:
        gif_images.append(imageio.imread(path))
    imageio.mimsave(gif_path + r'\test.gif', gif_images, fps=5)
    print(str(len(gif_images)) + ' GIF file was saved.')


def mkpath(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert("RGB")
    return img


def get_image(frames_number, shared_raw_list, shared_radar_running_flag, img_matrix):
    global index_of_current_frame, N_chirp, N_sample, N_frame, chirp_to_chirp_time_100ps, dist
    # 参数设置
    # Data split
    N_sample = 128
    N_chirp = 32
    N_frame = frames_number
    frame_period_us = 41666
    # 这个frame频率应该对测量精度没有任何影响，只是刷新率的问题；但是，它需要保证能够包含所有的chirp的time，即应该和gui中的frame hz有关，这个值设置小了就会报错65556
    lower_frequency_kHz = 58250000
    upper_frequency_kHz = 62930000

    chirp_to_chirp_time_100ps = 6194100  # 就是pulse repetion，其实它是和n_chirp完全对应的呀，然后限制了frame的频率

    chirp_time_in_gui = 133

    # Range FFT
    fs = 1000000  # 采样频率1MHz，上下频率算出来的
    n = np.arange(0, N_sample, 1)
    f = n * fs / N_sample
    c = 3 * 10 ** 8  # 电磁波传播速度
    # S = 3 * 10 ** 9 / (133 * 10 ** (-6))  # 扫频斜率 60-61 GHz，就是求tan得到的，chirp时间是不含间隔的
    S = (upper_frequency_kHz - lower_frequency_kHz) * 10 ** 3 / (
            chirp_time_in_gui * 10 ** (-6))  # 扫频斜率 60-61 GHz，就是求tan得到的，chirp时间是不含间隔的
    # 133是chirp time
    dist = f * c / (2 * S)



    # dist = f * c / (4 * S) #这里似乎必须是4倍，这和大多数的理论推导结果是不一样的，需要后续研究
    lock = Lock()
    # manager = Manager()
    # shared_list = manager.list()
    # shared_radar_running_flag = manager.list([0])
    # img_matrix = manager.list([[], [], []])
    p1 = Process(target=get_raw_signal_stream, args=(shared_raw_list, shared_radar_running_flag, N_sample, N_chirp,
                                                     fs, frame_period_us, lower_frequency_kHz,
                                                     upper_frequency_kHz, chirp_to_chirp_time_100ps,
                                                     lock))
    p2 = Process(target=handle_signal2img_process, args=(shared_raw_list, shared_radar_running_flag, img_matrix,
                                                         N_sample, N_chirp, chirp_to_chirp_time_100ps,
                                                         dist, lock))
    p1.start()
    p2.start()
    print('signal capture process have been started.')
    # while True:
    #     pass
    # print(shared_list)
    # p1.join()
    # p2.join()

    # signal_matrix = get_img_with_param(dev, N_frame)
    # img_matrix = draw_images(shared_list, shared_list_capa, lock)

    # return img_matrix


def get_raw_signal_stream(shared_raw_list, shared_radar_running_flag, N_sample, N_chirp, fs, frame_period_us,
                          lower_frequency_kHz, upper_frequency_kHz, chirp_to_chirp_time_100ps,
                          lock):
    global index_of_current_frame
    devconf = mkDevConf(num_samples_per_chirp=N_sample,
                        num_chirps_per_frame=N_chirp,
                        adc_samplerate_hz=fs,
                        frame_period_us=frame_period_us,
                        lower_frequency_kHz=lower_frequency_kHz,
                        upper_frequency_kHz=upper_frequency_kHz,
                        chirp_to_chirp_time_100ps=chirp_to_chirp_time_100ps,
                        )
    dev = ifx_device_create(devconf)
    frame = ifx_device_create_frame_from_device_handle(dev)


    # todo 循环50个4帧
    for repet in range(800):
        # print('Start your action!!!!')
        index_of_current_frame = 0
        # A loop for fetching a finite number of frames comes next..
        # make this a while loop in a try/except block to fetch frames indefinitely,
        # Ctrl+C to stop and disconnect from device.
        # Warning: ctrl+c might not work on Windows. If ifx_device_destroy is not
        # called, connecting to the device again might fail
        # try:
        # pre_time = -1
        # todo 这里暂时使用每一次4个frame
        for n in range(4):
            err = ifx_device_get_next_frame(dev, frame)
            if err == IFX_ERROR_FIFO_OVERFLOW:
                print("Got FIFO overflow")
                continue
            elif err == IFX_ERROR_TIMEOUT:
                continue
            else:
                index_of_current_frame += 1

            # Do some processing with the obtained frame.
            # In this example we just dump it into the console
            # if (pre_time != -1):
            #     frequency_of_frames = 1 / (time.time() - pre_time)
            #     pre_time = time.time()
            # else:
            #     frequency_of_frames = 0
            #     pre_time = time.time()
            # # print("Got frame " + format(frames) + ", num_antennas={}".format(frame.num_rx) + " f_frames: " + str(
            # #     frequency_of_frames) + "Hz")
            dem = np.zeros((3, N_sample * N_chirp))  # 这里的8192就是128*64

            for iAnt in range(0, frame.num_rx):
                mat = frame.per_antenna(iAnt)
                temp_d = np.ctypeslib.as_array(mat.data, shape=(mat.rows * mat.columns,))
                dem[iAnt, :] = temp_d

            # print(dem)

            # 把RX的信号提出来成一维数组
            if index_of_current_frame == 1:
                dem_1 = np.array(dem[0, :])
                dem_2 = np.array(dem[1, :])
                dem_3 = np.array(dem[2, :])
            else:
                dem1_temp = np.array(dem[0, :])
                dem_1 = np.append(dem_1, dem1_temp)
                dem2_temp = np.array(dem[1, :])
                dem_2 = np.append(dem_2, dem2_temp)
                dem3_temp = np.array(dem[2, :])
                dem_3 = np.append(dem_3, dem3_temp)

        with lock:
            # print('signal frame number is :' + str(repet))
            if len(shared_raw_list) == 0:
                # print('shared_raw_list = 0')
                shared_raw_list.append(dem_1)
                shared_raw_list.append(dem_2)
                shared_raw_list.append(dem_3)
                time.sleep(5)
                shared_radar_running_flag[0] = 1  # 0未开始，1开始，3结束
                print('system start.')
            else:
                shared_raw_list[0] = np.append(shared_raw_list[0], dem_1)
                shared_raw_list[1] = np.append(shared_raw_list[1], dem_2)
                shared_raw_list[2] = np.append(shared_raw_list[2], dem_3)

    shared_radar_running_flag[0] = 3
    close_device(dev)
    print('closing success.')


def take_raw_signal_to_img(N_chirp, N_sample, chirp_to_chirp_time_100ps, dist, img_matrix, lock,
                           signal_matrix, last_data_rd_fft):
    number_of_ann = 3
    take_data = [[], [], []]

    while len(signal_matrix[0]) < 4 * N_chirp * N_sample:
        # print('wait raw signals...')
        continue

    with lock:
        # print('raw signals lenth is ' + str(len(signal_matrix[0])))
        for ann in range(number_of_ann):
            take_data[ann] = signal_matrix[ann][0:4 * N_chirp * N_sample]  # 取出现有的信号
            # signal_matrix[ann][0:3] = np.array([])  # 将取出信号的位置删去
            signal_matrix[ann] = signal_matrix[ann][4 * N_chirp * N_sample:-1]  # 将取出信号的位置删去
    # print('raw signals have been take.')


    data_raw_1 = take_data[2].reshape(4, N_chirp, N_sample)
    data_raw_2 = take_data[0].reshape(4, N_chirp, N_sample)
    data_raw_3 = take_data[1].reshape(4, N_chirp, N_sample)

    frames_in_buffer = 4  # 固定4帧来进行signal2img

    ##### todo 处理第一通道数据，并且获得第一通道的相位
    data1_rd_fft_real = np.zeros((frames_in_buffer, N_chirp, N_sample))
    # data1_rd_fft_real = np.concatenate((last_data_rd_fft[0], data1_rd_fft_real), axis=0)
    data1_rd_fft_complex = np.zeros_like(data1_rd_fft_real, dtype=complex)
    data1_rd_fft_maskfiltered = np.zeros((frames_in_buffer, N_chirp, N_sample))

    img_list1 = []
    norm = 10
    for i in range(0, frames_in_buffer):
        data_temp1 = data_raw_1[i, :, :]
        data1_rd_fft_complex[i, :, :] = np.fft.fft2(data_temp1)
        # 正负速度拼接
        data1_rd_fft_complex[i, :, :] = np.r_[
            data1_rd_fft_complex[i, N_chirp // 2:N_chirp, :], data1_rd_fft_complex[i, 0:N_chirp // 2, :]]

        data1_rd_fft_real[i, :, :] = np.abs(data1_rd_fft_complex[i, :, :])

        data1_rd_fft_maskfiltered[i, :, :] = data1_rd_fft_real[i, :, :] - last_data_rd_fft[0]
        # 计算平均值
        last_data_rd_fft[1] = last_data_rd_fft[1] + 1
        last_data_rd_fft[0] = (data1_rd_fft_real[i, :, :] - last_data_rd_fft[0])/last_data_rd_fft[1] \
                              + last_data_rd_fft[0]
        # print('mean = ' + str(np.mean(last_data_rd_fft[0])))

        # # 强度过滤及归一化，或比例过滤
        data1_rd_fft_maskfiltered[i, :, :][data1_rd_fft_maskfiltered[i, :, :] > norm] = norm
        data1_rd_fft_maskfiltered[i, :, :] = data1_rd_fft_maskfiltered[i, :, :] / norm
        data1_rd_fft_maskfiltered[i, :, :][data1_rd_fft_maskfiltered[i, :, :] < 0.2] = 0

        # # 强度过滤及归一化，或比例过滤
        # data1_rd_fft_maskfiltered[i, :, :][data1_rd_fft_maskfiltered[i, :, :] < 1] = 0
        # max_value = np.max(data1_rd_fft_maskfiltered[i, :, :])
        # if max_value > 0:
        #     data1_rd_fft_maskfiltered[i, :, :] = data1_rd_fft_maskfiltered[i, :, :] / max_value
        #     data1_rd_fft_maskfiltered[i, :, :][data1_rd_fft_maskfiltered[i, :, :] < 0.25] = 0

            # data_rd_fft_maskfiltered[data_rd_fft_maskfiltered < 0.25 * max_value] = 0

        # 屏蔽距离较远的动作，避免误识别(只剩了16点)
        data1_rd_fft_maskfiltered[i, :, N_sample // 8:-1] = 0

        image = torch.from_numpy(data1_rd_fft_maskfiltered[i, :, 0:32])
        image.squeeze(0)
        img_list1.append(image)  # img_list是当前信道的image序列
    # print('complete draw')
    img_list1 = torch.stack(img_list1, dim=0)

    if len(img_matrix[0]) == 0:
        img_matrix[0] = img_list1
    else:
        img_matrix[0] = torch.cat((img_matrix[0], img_list1), dim=0)


    ##### todo 处理第二通道数据，并且获得相位差
    img_list2 = []
    data2_rd_fft_complex = np.zeros((frames_in_buffer, N_chirp, N_sample), dtype=complex)
    data2_ra_fft_real = np.zeros((frames_in_buffer, N_chirp, N_sample))
    for i in range(0, frames_in_buffer):
        data_temp2 = data_raw_2[i, :, :]
        data2_rd_fft_complex[i, :, :] = np.fft.fft2(data_temp2)
        # for chirp_id in range(data_temp2.shape[0]):
        #     data2_r_fft_complex[i, chirp_id, :] = np.fft.fft(data_temp2[chirp_id, :])
        # for sample_id in range(data_temp2.shape[1]):
        #     data2_rd_fft_complex[i, :, sample_id] = np.fft.fft(
        #         data2_r_fft_complex[i, :, sample_id])

        data2_rd_fft_complex[i, :, :] = np.r_[
            data2_rd_fft_complex[i, N_chirp // 2:N_chirp, :], data2_rd_fft_complex[i,
                                                                          0:N_chirp // 2, :]]

        data2_ra_fft_real[i, :, :] = np.angle(data2_rd_fft_complex[i, :, :]) - np.angle(
            data1_rd_fft_complex[i, :, :])
        # data2_ra_fft_real[i, :, :] = data2_ra_fft_real[i, :, :] / np.max(np.max(data2_ra_fft_real[i, :, :]))

        data2_ra_fft_real[i, :, :] = np.arcsin((data2_ra_fft_real[i, :, :] / pi) % 2 - 1) / (
                    pi / 2)
        mask = (data1_rd_fft_maskfiltered[i, :, :] == 0)
        data2_ra_fft_real[i, :, :][mask] = 0

        # # 二值化
        # data2_ra_fft_real[data2_ra_fft_real > 0] = 1
        # data2_ra_fft_real[data2_ra_fft_real < 0] = -1

        image = torch.from_numpy(data2_ra_fft_real[i, :, 0:32])
        image.squeeze(0)
        img_list2.append(image)  # img_list是当前信道的image序列
    # print('complete draw')
    img_list2 = torch.stack(img_list2, dim=0)

    if len(img_matrix[1]) == 0:
        img_matrix[1] = img_list2
    else:
        img_matrix[1] = torch.cat((img_matrix[1], img_list2), dim=0)

    ##### todo 处理第三通道数据，并且获得相位差
    img_list3 = []
    data3_rd_fft_complex = np.zeros((frames_in_buffer, N_chirp, N_sample), dtype=complex)
    data3_ra_fft_real = np.zeros((frames_in_buffer, N_chirp, N_sample))
    for i in range(0, frames_in_buffer):
        data_temp3 = data_raw_3[i, :, :]
        data3_rd_fft_complex[i, :, :] = np.fft.fft2(data_temp3)
        # for chirp_id in range(data_temp3.shape[0]):
        #     data3_r_fft_complex[i, chirp_id, :] = np.fft.fft(data_temp3[chirp_id, :])
        # for sample_id in range(data_temp3.shape[1]):
        #     data3_rd_fft_complex[i, :, sample_id] = np.fft.fft(
        #         data3_r_fft_complex[i, :, sample_id])

        data3_rd_fft_complex[i, :, :] = np.r_[
            data3_rd_fft_complex[i, N_chirp // 2:N_chirp, :], data3_rd_fft_complex[i,
                                                              0:N_chirp // 2, :]]

        data3_ra_fft_real[i, :, :] = np.angle(data3_rd_fft_complex[i, :, :]) - np.angle(
            data1_rd_fft_complex[i, :, :])
        # data3_ra_fft_real[i, :, :] = data3_ra_fft_real[i, :, :] / np.max(np.max(data3_ra_fft_real[i, :, :]))

        data3_ra_fft_real[i, :, :] = np.arcsin((data3_ra_fft_real[i, :, :] / pi) % 2 - 1) / (
                pi / 2)
        mask = (data1_rd_fft_maskfiltered[i, :, :] == 0)
        data3_ra_fft_real[i, :, :][mask] = 0

        # # 二值化
        # data3_ra_fft_real[data3_ra_fft_real > 0] = 1
        # data3_ra_fft_real[data3_ra_fft_real < 0] = -1

        image = torch.from_numpy(data3_ra_fft_real[i, :, 0:32])
        image.squeeze(0)
        img_list3.append(image)  # img_list是当前信道的image序列
    # print('complete draw')
    img_list3 = torch.stack(img_list3, dim=0)

    if len(img_matrix[2]) == 0:
        img_matrix[2] = img_list3
    else:
        img_matrix[2] = torch.cat((img_matrix[2], img_list3), dim=0)

    return True


def handle_signal2img_process(shared_raw_list, shared_radar_running_flag, img_matrix, N_sample, N_chirp,
                              chirp_to_chirp_time_100ps, dist,
                              lock):
    while len(shared_raw_list) == 0:
        continue
    print('ok, raw signal has data.')
    # no_signal_count = 0
    frames_has_been_trans = 0
    process_count = 0
    # last_data_rd_fft = [np.zeros((2, N_chirp, N_sample)), np.zeros((2, N_chirp, N_sample)), np.zeros((2, N_chirp, N_sample))]
    last_data_rd_fft = [np.zeros((N_chirp, N_sample)), 0]
    while shared_radar_running_flag[0] != 3 or len(shared_raw_list[0]) != 0:
        while len(shared_raw_list) == 0:
            continue
        # print('ok, raw signal has data.')
        # print('this is ' + str(process_count + 1) + 'th signal2img process.')
        take_raw_signal_to_img(N_chirp, N_sample, chirp_to_chirp_time_100ps, dist,
                               img_matrix, lock, shared_raw_list, last_data_rd_fft)
        process_count += 1

    # print('signal2img process has been complete, total are ' + str(frames_has_been_trans) + ' frames')
    # print('still ' + str(len(signal_matrix[0])) + ' frames havnt been trans')
    return img_matrix


def close_device(dev):
    print("Closing the device")
    ifx_device_destroy(dev)
