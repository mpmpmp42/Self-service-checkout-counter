from time import time

import serial.tools.list_ports
import serial


def show_ports():
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print('找不到串口')
    else:
        for i in range(0, len(port_list)):
            print(port_list[i])


class scaleapi:
    weight = 0

    def __init__(self, ser):
        self.ser = ser

    def readweight(self):
        self.ser.write(b'\xA3\x00\xA2\xA4\xA5')
        p5 = self.ser.read(5)
        p6 = self.ser.read(6)
        digi5 = p5[-1]
        digi6 = p6[0]
        digi7 = int(p6.hex("-").split("-")[1], 16)
        self.weight = digi5 * 65536 + digi6 * 256 + digi7

        return self.weight


if __name__ == '__main__':
    # using example
    ser = serial.Serial('com5', 9600, timeout=0.5)
    scale1 = scaleapi(ser)
    while True:
        print(scale1.readweight())
