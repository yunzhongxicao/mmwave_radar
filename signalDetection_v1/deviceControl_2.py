from ctypes import *
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import axes3d
import time

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
def mkDevConf(num_samples_per_chirp = 128,
              num_chirps_per_frame = 64,
              adc_samplerate_hz = 1000000,
              # 那么，一个chirp的时常为num_samples_per_chirp/adc_samplerate_hz ~ 0.000133s
              frame_period_us = 150000,
              lower_frequency_kHz = 59700000,
              upper_frequency_kHz = 61700000,
              # 这里，上下相减就是扫频的带宽
              bgt_tx_power = 31,
              rx_antenna_mask = 7, #目前是111，则3个天线都打开（测试过不打开，所以应该是可以控制的）
              chirp_to_chirp_time_100ps = 5056400,
              # 这个就是chirp时间加上中间的间隔时间，也就是gui中的pulse repetition。间隔时间不可直接设置，间接来自于这个设置
              #32×1870000×100 + 400000000×100 = ps/10(12)=0.045984s
              if_gain_dB = 33,
              frame_end_delay_100ps = 400000000, #0.04s，每个frame之间的间隔
              shape_end_delay_100ps = 1500000): #这个值明显是被覆盖了的，而且以上两个值在frame层面，对信号采集已经没有品质上的影响了
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
IFX_OK = 0                       # No error
IFX_ERROR_TIMEOUT = 65559        # Timeout occurred on communication interface with device
IFX_ERROR_FIFO_OVERFLOW = 65560  # FIFO Overflow occurred, signifying potential loss of data 

# =====================================================================================
# Radar SDK API wrappers
# =====================================================================================
radar_sdk = CDLL('radar_sdk_dll')
radar_sdk.ifx_device_create.restype = c_int
radar_sdk.ifx_device_create.argtypes = [c_void_p , c_void_p]

radar_sdk.ifx_device_destroy.restype = c_int
radar_sdk.ifx_device_destroy.argtypes = [c_void_p]

radar_sdk.ifx_device_create_frame_from_device_handle.restype = c_int
radar_sdk.ifx_device_create_frame_from_device_handle.argtypes = [c_void_p , c_void_p]

radar_sdk.ifx_device_get_next_frame.restype = c_int
radar_sdk.ifx_device_get_next_frame.argtypes = [c_void_p , c_void_p]

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

    # 参数设置
    # Data split
    N_sample = 128
    N_chirp = 32
    N_frame = 200
    frame_period_us = 41666
    # 这个frame频率应该对测量精度没有任何影响，只是刷新率的问题；但是，它需要保证能够包含所有的chirp的time，即应该和gui中的frame hz有关，这个值设置小了就会报错65556
    lower_frequency_kHz = 58250000
    upper_frequency_kHz = 62930000

    chirp_to_chirp_time_100ps = 6194100 # 就是pulse repetion，其实它是和n_chirp完全对应的呀，然后限制了frame的频率

    chirp_time_in_gui = 133

    # Range FFT
    fs = 2000000  # 采样频率1MHz，上下频率算出来的
    n = np.arange(0, N_sample, 1)
    f = n * fs / N_sample
    c = 3 * 10 ** 8  # 电磁波传播速度
    # S = 3 * 10 ** 9 / (133 * 10 ** (-6))  # 扫频斜率 60-61 GHz，就是求tan得到的，chirp时间是不含间隔的
    S = (upper_frequency_kHz - lower_frequency_kHz) * 10 ** 3/ (chirp_time_in_gui * 10 ** (-6))  # 扫频斜率 60-61 GHz，就是求tan得到的，chirp时间是不含间隔的
    # 133是chirp time
    dist = f * c / (2 * S)
    # dist = f * c / (4 * S) #这里似乎必须是4倍，这和大多数的理论推导结果是不一样的，需要后续研究

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

    frames = 0


    # A loop for fetching a finite number of frames comes next..
    # make this a while loop in a try/except block to fetch frames indefinitely, 
    # Ctrl+C to stop and disconnect from device. 
    # Warning: ctrl+c might not work on Windows. If ifx_device_destroy is not
    # called, connecting to the device again might fail    
    #try:
    pre_time = -1
    for n in range(N_frame):
            err = ifx_device_get_next_frame(dev, frame)
            if err == IFX_ERROR_FIFO_OVERFLOW:
                print ("Got FIFO overflow")
                continue
            elif err == IFX_ERROR_TIMEOUT:
                continue
            else:
                frames += 1
            
            # Do some processing with the obtained frame.
            # In this example we just dump it into the console
            if(pre_time!=-1):
                f_frame = 1/(time.time()-pre_time)
                pre_time = time.time()
            else:
                f_frame = 0
                pre_time = time.time()
            print ("Got frame " + format(frames) + ", num_antennas={}".format(frame.num_rx) + " f_frames: " +str(f_frame) + "Hz")
            dem=np.zeros((3,N_sample*N_chirp)) #这里的8192就是128*64
            
            for iAnt in range(0, frame.num_rx):
                mat = frame.per_antenna(iAnt)
                temp_d=np.ctypeslib.as_array(mat.data,shape=(mat.rows*mat.columns,))
                dem[iAnt,:]=temp_d
                
            print(dem)

            # 把2号RX的信号提出来成一维数组
            if(frames == 1):
                dem_2 = dem_temp = np.array(dem[1, :])
            else:
                dem_temp = np.array(dem[1, :])
                dem_2 = np.append(dem_2, dem_temp)

    # plt.plot(dem[2, 0:2048])
    # plt.show()
    #
    # axisX = np.linspace(0,0.005984,2048) #这样，chirp到chirp之间的间隔是被忽略掉了的吧
    # sampling_rate = 2000000
    # fft_size = 512
    # y_fft_2 = fft(dem[2, 0:fft_size]) / len(dem[2, 0:fft_size])
    # freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
    #
    # #计算双侧频谱,后基于信号长度L=512,采样频率2MHz,计算单侧频谱
    # y_fft_2_amp = np.abs(y_fft_2)
    # y_fft_2_amp_half = y_fft_2_amp[0:256]
    # #yFF2_1 =2*yFF2_1
    #
    # #定义频域f并绘制单边频谱
    # # f = (2/512)*np.arange(0,256,1)
    # f = freqs
    # #距离
    # d = 1.11*2/512*np.arange(0,256,1)
    #
    #
    # #xFF0 = np.arange(len(axisX))
    # plt.subplot(211)
    # plt.title("0.2m test")
    # plt.xlabel("Samples")
    # plt.ylabel("A ")
    # plt.plot(dem[2,0:512])
    # plt.subplot(212)
    # #plt.title("Rx2")
    # plt.xlabel("Frequency/MHz")
    # plt.ylabel("A ")
    # plt.plot(f, y_fft_2_amp_half)
    # plt.show()
    #
    # '''
    # plt.subplot(313)
    # #plt.title("Rx3")
    # plt.xlabel("Samples")
    # plt.ylabel("A ")
    # plt.plot(dem[2,0:512])
    # plt.show()
    # '''

    # except KeyboardInterrupt:

    dem_2 = dem_2.reshape(frames, N_chirp, N_sample)
    data = dem_2
    # print(dem_2)


    data_fft = np.zeros((N_frame, N_chirp, N_sample), dtype=complex)

    for i in range(0, N_frame):
        for j in range(0, N_chirp):
            # data_temp = data[i, j, :] = data[i, j, :] - np.mean(data[i, j, :])
            data_temp = data[i, j, :]
            data_fft[i, j, :] = fft(data_temp, N_sample)

    fig1 = plt.figure()
    ax = fig1.add_subplot(211)
    ax.plot(dist, dem_2[0, 0, :])
    ax = fig1.add_subplot(212)
    ax.plot(dist, np.abs(data_fft[0, 0, :])/len(data_fft[0, 0, :]))
    # ax = fig1.add_subplot(412)
    # ax.plot(dist, data_fft[0, 1, :])
    # ax = fig1.add_subplot(413)
    # ax.plot(dist, data_fft[1, 0, :])
    # ax = fig1.add_subplot(414)
    # ax.plot(dist, data_fft[1, 1, :])
    plt.show()
    print("The most important distance: " + str(dist[np.argmax(len(data_fft[0, 0, :]) * np.abs(data_fft[0, 0, :]))]))
    # np.savetxt('20200805_23_17_data_fft.txt',np.abs(data_fft[0,:,:]))

    # Range Doppler
    fc = int(1 / (chirp_to_chirp_time_100ps * 10 ** (-10)))  # Tc=505.64 μs,
    n_chirp = np.arange(-N_chirp // 2, N_chirp // 2 + 1, 1)
    f_chirp = n_chirp * fc / N_chirp
    v_chirp = 2.5 * 10 ** (-3) * f_chirp
    # 这里忘了1是什么原因了，是某个自己设置的放大倍数？？？

    data_RD_fft = np.zeros((N_frame, N_chirp, N_sample), dtype=complex)
    data_RD_fft_difference = data_RD_fft.copy()
    data_speed = np.zeros((N_frame, 1), dtype=float)
    peak_angle = np.zeros((N_frame, 1), dtype=float)
    data_RD_fft_sum = np.zeros((N_chirp, N_sample), dtype=complex)

    for i in range(0, N_frame):
        for j in range(0, N_sample):
            # data_fft_temp = data_fft[i, :, j] - np.mean(data_fft[i, :, j])
            data_fft_temp = data_fft[i, :, j]
            data_RD_fft[i, :, j] = fft(data_fft_temp, N_chirp)

        pos = np.unravel_index(np.argmax(data_RD_fft[i, 0:int(N_chirp / 2), 0:int(N_sample / 2)]),
                               np.shape(data_RD_fft[i, :, :]))  # 查找最大的目标点

        # data_speed[i] = v_chirp[pos[0]]
        peak_angle[i] = np.angle(data_RD_fft[i, pos[0], pos[1]])

        ##累计速度求和
        # data_RD_fft_sum = data_RD_fft_sum + data_RD_fft[i,:,:]

        # Range Doppler 正负速度拼接,
        data_RD_fft[i, :, :] = np.r_[data_RD_fft[i, N_chirp // 2:N_chirp, :], data_RD_fft[i, 0:N_chirp // 2, :]]

        if i != 0:
            data_RD_fft_difference[i, :, :] = data_RD_fft[i, :, :] - data_RD_fft[i - 1, :, :]
            # data_RD_fft_difference[i, :, :] = data_RD_fft[i, :, :] - data_RD_fft[0, :, :]
            # data_RD_fft_difference[i, :, :] = data_RD_fft[i, :, :]
        else:
            data_RD_fft_difference[i, :, :] = data_RD_fft[i, :, :]

        # 显示 Range Doppler
        plt.subplot(1, 1, 1)
        pcm = plt.pcolormesh(dist[0:int(N_sample/2)], v_chirp, np.abs(data_RD_fft_difference[i, :, 0:int(N_sample/2)]),
                             cmap='plasma', vmin=0.0, vmax=5)  # 'PuBu_r'
        # pcm = plt.pcolormesh(dist[0:int(N_sample / 2)//2], v_chirp[0:int(N_chirp / 2)//2],
        #                      np.abs(data_RD_fft[i, 0:int(N_chirp / 2)//2, 0:int(N_sample / 2)//2]), cmap='plasma',
        #                      vmin=0.0, vmax=15)  # 'PuBu_r'
        plt.colorbar(pcm)
        plt.title("Figure %d " % (i))
        plt.xlabel("Range - m")
        plt.ylabel("Doppler V - m/s")
        # plt.ylim(0, 5)
        # plt.xlim(0, 2)
        # 显示 Range 峰值对应的相位
        # plt.subplot(2, 1, 2)
        # plt.plot(n_chirp, np.angle(data_fft[i, :, pos[1]]))
        # plt.xlabel("chirp ID")
        # plt.ylabel("Peak phase angle")
        plt.draw()
        # plt.pause(1)
        plt.savefig(r'C:\Users\bitwa\Documents\MMW_radar_test\test\Figure %d.jpg' % (i + 1))
        plt.close()

    # Range - Doppler 累加求和
    '''
    pcm_sum = plt.pcolormesh(dist[0:int(N_sample/2)],v_chirp[0:int(N_chirp/2)],np.abs(data_RD_fft_sum[0:int(N_chirp/2), 0:int(N_sample/2)]),cmap = 'plasma')
    plt.colorbar(pcm_sum)
    plt.title("Figure sum ")
    plt.xlabel("Range - m")
    plt.ylabel("Doppler V - m/s")
    '''

    # np.savetxt('20200805_23_17_data_RangeDoppler_fft.txt',np.abs(data_RD_fft[0,:,:]))

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

    print("Closing the device")
    ifx_device_destroy(dev)
