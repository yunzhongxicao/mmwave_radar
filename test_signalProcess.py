"""
@Desc   :这里用于测试多组数据同时进行处理，简化输入流程
"""
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from signalDetection_v2.get_raw_data import radar_data_read, de, find_peak,radar_data_write
from signalDetection_v2.read_raw_data import signalProcess
import glob, os

# root_path = r'E:\基于毫米波的运动监测\device_data\2021_03_03_measure\3GHz'
# file_path = glob.glob(os.path.join(root_path, '*.npy'))
# para_path = glob.glob(os.path.join(root_path, '*.txt'))
# print(file_path)
# print(para_path[0])
# print()
# for f in file_path:
#     signalProcess(f, para_path[0])

root_path = 'E:\\基于毫米波的运动监测\\PythonWrapper_Win\\signalDetection_v3\\2021_03_25_measure\\'
para_path = root_path + 'para.txt'
data_path = root_path + "test.npy"

radar_data_write(upper_frequency_kHz=63000000,num_frame=50,data_path=data_path,para_path=para_path)
signalProcess(data_path,para_path)

