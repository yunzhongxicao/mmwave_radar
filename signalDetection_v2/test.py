import pandas as pd
import numpy as np
import datetime

a = np.zeros((3, 2))
print(a)
path = 'E:\\基于毫米波的运动监测\\PythonWrapper_Win\\device_data\\'

print(datetime.datetime.now())
year_int = datetime.datetime.now().year
month_int = datetime.datetime.now().month
day_int = datetime.datetime.now().day
min_int = datetime.datetime.now().minute
second_int = datetime.datetime.now().second
nowtime = str(year_int) + '_'+ str(month_int) + '_'+ str(day_int) + '_' + str(min_int) + '_'+ str(second_int)
file_path = path + nowtime + '.txt'
file = open(file_path, 'w')
b = 2345
file.write('123' + '=' + str(b) +'\n')
file.write('abd'+'\n')
file.write('num_samples_per_chirp=64'+'\n')
file.write('num_chirps_per_frame=32'+'\n')

data_path = path + nowtime + 'datatest.csv'
a_df = pd.DataFrame(a)
a_df.to_csv(data_path,index=False)

b = pd.read_csv('../device_data/2021_1_16_38_35_data.csv')
print(b.head())
b = np.array(b)
print(b.shape)

para = pd.read_table('../device_data/2021_1_16_38_35_para.txt',header=None)
print(para)
para = np.array(para)
print(para.shape)
para = para[:,0]
print(para.shape)

for i in range(4):
    para[i] = para[i].split('=')

print(para[0][1])

