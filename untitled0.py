import serial

import time

ser = serial.Serial('COM5', 9600, bytesize=8, parity='N', stopbits=1, timeout=2)

# ser.write('1'.encode())
# print(ser.readlines())
# ser.write('1'.encode())
# print(ser.readlines())


# ser.write('4'.encode()) 


# ser.write('5'.encode()) 

# while 1:
#     s = ser.readline()
#     print(s)
#     if s=='OVER\r\n'.encode('utf_8'):
#         break


ser.write('2'.encode())
print(ser.readlines())

ser.close()
time.sleep(0.1)

