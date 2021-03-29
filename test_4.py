'''
@File  :test_4.py
@Author:dfc
@Date  :2021/3/117:35
@Desc  :
'''
import numpy as np
import matplotlib.pyplot as plt
a = np.array([1, 4, 3, -1, 6])
print(a[2:])
print(np.argmax(a))

b = np.array([1, 4, 3, -1, 6])
plt.plot(a,b)
plt.xlim(1,)
plt.show()

