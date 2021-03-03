from ctypes import *
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft


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
              # chirp_to_chirp_time_100ps=1870000,  # 32×1870000×100 + 400000000×100= ps/10(12)=0.045984s
              chirp_to_chirp_time_100ps=0,
              if_gain_dB=33,  # IF信号放大系数
              frame_end_delay_100ps=400000,
              shape_end_delay_100ps=400000):
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


# =====================================================================================
# Example Usage
# =====================================================================================

if __name__ == "__main__":
    devconf = mkDevConf()
    dev = ifx_device_create(devconf)
    frame = ifx_device_create_frame_from_device_handle(dev)

    frames = 0

    # A loop for fetching a finite number of frames comes next..
    # make this a while loop in a try/except block to fetch frames indefinitely,
    # Ctrl+C to stop and disconnect from device.
    # Warning: ctrl+c might not work on Windows. If ifx_device_destroy is not
    # called, connecting to the device again might fail
    # try:
    for n in range(10):
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
        print("Got frame " + format(frames) + ", num_antennas={}".format(frame.num_rx))
        dem = np.zeros((3, 2048))

        for iAnt in range(0, frame.num_rx):
            mat = frame.per_antenna(iAnt)
            temp_d = np.ctypeslib.as_array(mat.data, shape=(mat.rows * mat.columns,))
            dem[iAnt, :] = temp_d
        print(dem.shape)
        print(dem)

    L = 64
    Fs = 2000000

    # standard
    m = dem.mean(axis=1)
    print('dem shape', dem.shape)
    print(m.shape)
    m = m.reshape(3, -1)
    print(m.shape)
    dem = dem - m

    # fft
    sp = np.fft.fft(dem[2, 0:L])
    P2 = np.abs(sp / L) * 2  # 双边谱 且对称 归一化
    P1 = P2[range(int(L / 2))]  # 取一半

    # 	定义频谱
    f = np.arange(int(L / 2)) * Fs / L

    # 转化为距离
    c = 3 * (10 ** 8)  # 光速
    S = (devconf.upper_frequency_kHz - devconf.lower_frequency_kHz) / (
            devconf.chirp_to_chirp_time_100ps - devconf.shape_end_delay_100ps)
    S = S * 1000 / (100 * (10 ** (-12)))
    print(S)
    print(c / 2 / S)

    d = f * c / 2 / S

    plt.plot(f, P1)
    plt.show()

    plt.plot(d, P1)
    plt.show()
