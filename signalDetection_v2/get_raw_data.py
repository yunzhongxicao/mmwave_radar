from ctypes import *
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
import datetime
import pandas as pd


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
class Frame(Structure):  # 完整的一个时域帧
    _fields_ = [('num_rx', c_uint8),
                ('rx_data', POINTER(Matrix))]

    ## all_antenna Method
    # Returns the pointer to the beginning of the data of the entire frame
    # @param self      The pointer to the Frame
    def all_antenna(self):  # 所有帧的方法
        return self.rx_data

    ## per_antenna Method
    # Returns the pointer to the beginning of the data of the the specified antenna
    # @param self      The pointer to the Frame
    # @param antenna   Number of the antenna whose radar is required
    def per_antenna(self, antenna):  # 每一帧的方法
        return self.rx_data[antenna]


# =====================================================================================
## mkDevConf Method
# This method returns a DeviceConfig Structure, with its fields having0 the values
# specified, or if not specified, having default values.
# =====================================================================================
def mkDevConf(num_samples_per_chirp=64,  # 每一个chirp采样数
              num_chirps_per_frame=32,  # 每一帧chirp数目
              adc_samplerate_hz=2000000,  # adc采样率
              frame_period_us=0,  # 两个连续帧之间的时间间隔
              lower_frequency_kHz=58000000,  # FMCW低频
              upper_frequency_kHz=63000000,  # FMCW高频
              bgt_tx_power=31,  # TX信号功率值，0-31
              rx_antenna_mask=7,  # 接受天线
              chirp_to_chirp_time_100ps=1870000,  # 32×1870000×100 + 400000000×100= ps/10(12)=0.045984s
              if_gain_dB=33,  # IF信号放大系数
              frame_end_delay_100ps=400000000,  # 每个frame之间的间隔，0.04s
              shape_end_delay_100ps=1500000):  # 这两个值是被覆盖了的，对信号采集没有明显影响
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
radar_sdk = CDLL('../radar_sdk_dll')
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


# ==================================================================================
# 读取并存储雷达得到的数据
# 输入为各项参数
# 得到的为路径上的两个文件，一个csv为雷达数据，row是所有chirp堆起来，column是一个chirp3个天线的采样数据
# txt为雷达参数
# 两个输出文件命名均为当前时间
# ==================================================================================
def radar_data_write(num_samples_per_chirp=128,
                     num_chirps_per_frame=512,
                     adc_samplerate_hz=2000000,
                     frame_period_us=0,
                     lower_frequency_kHz=58000000,
                     upper_frequency_kHz=63000000,
                     bgt_tx_power=31,
                     rx_antenna_mask=7,
                     chirp_to_chirp_time_100ps=0,  # 32×1870000×100 + 400000000×100= ps/10(12)=0.045984s
                     if_gain_dB=33,
                     frame_end_delay_100ps=1500000,
                     shape_end_delay_100ps=1500000,
                     num_frame=100):
    chirp_time_in_gui = 133  # 设置完成其他参数后从GUI读取得到的
    devconf = mkDevConf(num_samples_per_chirp, num_chirps_per_frame, adc_samplerate_hz, frame_period_us,
                        lower_frequency_kHz, upper_frequency_kHz, bgt_tx_power,
                        rx_antenna_mask, chirp_to_chirp_time_100ps, if_gain_dB, frame_end_delay_100ps,
                        shape_end_delay_100ps)
    dev = ifx_device_create(devconf)
    frame = ifx_device_create_frame_from_device_handle(dev)

    frames = 0
    dem = np.zeros((num_chirps_per_frame * num_frame, num_samples_per_chirp * 3))
    for n in range(num_frame):
        err = ifx_device_get_next_frame(dev, frame)
        if err == IFX_ERROR_FIFO_OVERFLOW:
            print("Got FIFO overflow")
            continue
        elif err == IFX_ERROR_TIMEOUT:
            continue
        else:
            frames += 1

        # Do some processing with the obtained frame.
        # In this example we just dump it into the console
        print("Goft frame " + format(frames) + ", num_antennas={}".format(frame.num_rx))

        for iAnt in range(0, frame.num_rx):
            mat = frame.per_antenna(iAnt)
            temp_d = np.ctypeslib.as_array(mat.data, shape=(mat.rows, mat.columns))
            a = np.shape(temp_d)[0]
            b = np.shape(temp_d)[1]
            print(a, b)

            # dem[n * a: (n + 1) * a, iAnt * b:(iAnt + 1) * b] = temp_d
            dem[(n * np.shape(temp_d)[0]): ((n + 1) * np.shape(temp_d)[0]),
            (iAnt * np.shape(temp_d)[1]): ((iAnt + 1) * np.shape(temp_d)[1])] = temp_d

    # save para and data
    root_path = 'E:\\基于毫米波的运动监测\\device_data\\'
    year_int = datetime.datetime.now().year
    month_int = datetime.datetime.now().month
    day_int = datetime.datetime.now().day
    hour_int = datetime.datetime.now().hour
    min_int = datetime.datetime.now().minute
    second_int = datetime.datetime.now().second
    nowtime = str(year_int) + '_' + str(month_int) + '_' + str(day_int) + '_' + str(hour_int) + '_' + str(
        min_int) + '_' + str(second_int)
    para_path = root_path + nowtime + '_para.txt'
    para_file = open(para_path, 'w')
    para_file.write('num_samples_per_chirp=' + str(num_samples_per_chirp) + '\n')
    para_file.write('num_chirps_per_frame=' + str(num_chirps_per_frame) + '\n')
    para_file.write('adc_samplerate_hz=' + str(adc_samplerate_hz) + '\n')
    para_file.write('frame_period_us=' + str(frame_period_us) + '\n')
    para_file.write('lower_frequency_kHz=' + str(lower_frequency_kHz) + '\n')
    para_file.write('upper_frequency_kHz=' + str(upper_frequency_kHz) + '\n')
    para_file.write('bgt_tx_power=' + str(bgt_tx_power) + '\n')
    para_file.write('rx_antenna_mask=' + str(rx_antenna_mask) + '\n')
    para_file.write('chirp_to_chirp_time_100ps=' + str(chirp_to_chirp_time_100ps) + '\n')
    para_file.write('if_gain_dB=' + str(if_gain_dB) + '\n')
    para_file.write('frame_end_delay_100ps=' + str(frame_end_delay_100ps) + '\n')
    para_file.write('shape_end_delay_100ps=' + str(shape_end_delay_100ps) + '\n')
    para_file.write('num_frame=' + str(num_frame) + '\n')
    para_file.write('chirp_time_in_gui=' + str(chirp_time_in_gui) + '\n')

    data_path = root_path + nowtime + '_data.npy'
    np.save(data_path, dem)

    print("Closing the device")
    ifx_device_destroy(dev)


# =====================================================================================
# Example Usage
# =====================================================================================

if __name__ == "__main__":
    radar_data_write(num_frame=100)

