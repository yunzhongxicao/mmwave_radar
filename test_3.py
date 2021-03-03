# from scipy import integrate
#
# x2 = lambda x: x ** 2
# f = lambda x, a: a * x
# y, err = integrate.quad(f, 0, 1, args=(1,))
# print(y)

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
from signalDetection_v2.get_raw_data import radar_data_read

print(sys.path)

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
x = np.array([1, 4, 3, -1, 6])
plt.plot(x)
plt.title("测试",fontproperties= font)
plt.show()
print(x[x.argsort()[-3:]])
print(x.argsort())
sp = np.fft.fft(x)
sp = (sp - sp.mean()) * 2 / 5
sp[0] = sp[0] / 2
x_fft = np.abs(sp)
print(sp)
print(x_fft)
print(x_fft.imag)

y = x[-2:]
print(len(y))

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.power(a, 2)
print(a)
print(b)

c = np.sum(a, axis=0)
print(c)
print(c.shape)

d = c[::-1]
print(d)
