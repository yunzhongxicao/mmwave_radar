"""
@File  :test_arduino_serial.py
@Author:dfc
@Date  :2021/3/1220:25
@Desc  :这里只是测试利用python进行串口通信可行性
"""
import serial

# 打开串口
serialPort = "COM6"
baudRate = 9600
ser = serial.Serial(serialPort, baudRate, timeout=0.5)

demo1 = b'0'
demo2 = b'1'
demo3 = b'2'

while 1:
    c = input('请输入指令：')
    c = int(c)
    if c == 0:
        ser.write(demo1)
    if c == 1:
        ser.write(demo2)
    if c == 2:
        ser.write(demo3)
