import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from signalDetection_v2.get_raw_data import radar_data_read, de, find_peak
from signalDetection_v2.read_raw_data import signalProcess
import glob, os

root_path = r'E:\基于毫米波的运动监测\device_data\2021_03_03_measure\3GHz'
file_path = glob.glob(os.path.join(root_path, '*.npy'))
print(file_path)
print()
for f in file_path:
    signalProcess(f,'E:\\基于毫米波的运动监测\\device_data\\2021_03_03_measure\\3GHz\\para.txt')
