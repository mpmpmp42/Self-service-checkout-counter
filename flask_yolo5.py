# -*- coding: utf-8 -*-#
import serial
from flask import Flask, render_template, request, jsonify, make_response
from datetime import timedelta
import os
import detectapi
import cv2
import scale

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

ser = serial.Serial('com5', 9600, timeout=0.5)
scale1 = scale.scaleapi(ser)

det = detectapi.DetectAPI()
det.loadmodel()

video_capture = cv2.VideoCapture(1)
ret, frame = video_capture.read()

regular_result = []
total_prices = 0


def regu(result):
    global total_prices
    labels = []

    for dicti in result:

        if dicti['label'] not in ['banana_wb', 'banana_wob', 'blackberries', 'raspberry', 'lemon_wb',
                                  'lemon_wob', 'grapes_wb', 'grapes_wob',
                                  'tomato_wb', 'tomato_wob', 'apple_wb', 'apple_wob', 'chilli_wb',
                                  'chilli_wob']:
            labels.append(dicti['label'])
            regular_result.append(dicti)

        if dicti['label'] in ['banana_wb', 'banana_wob', 'blackberries', 'raspberry', 'lemon_wb', 'lemon_wob',
                              'grapes_wb', 'grapes_wob',
                              'tomato_wb', 'tomato_wob', 'apple_wb', 'apple_wob', 'chilli_wb', 'chilli_wob']:
            dicti['number'] = str(scale1.readweight()) + 'g'
            labels.append(dicti['label'])
            regular_result.append(dicti)

    for i in range(len(regular_result)):
        if regular_result[i]['label'] == 'banana_wob':
            regular_result[i]['label'] = 'Banana'
            price = round(int(regular_result[i]['number'][:-1]) * 0.00548, 2)
            total_prices += price
            regular_result[i]['total_prices'] = str(price) + 'Yuan'

        if regular_result[i]['label'] == 'apple_wob':
            regular_result[i]['label'] = 'Apple'
            price = round(int(regular_result[i]['number'][:-1]) * 0.013, 2)
            total_prices += price
            regular_result[i]['total_prices'] = str(price) + 'Yuan'

        if regular_result[i]['label'] == 'haoliyou':
            regular_result[i]['label'] = 'ORION PIE'
            price = int(regular_result[i]['number']) * 5
            total_prices += price
            regular_result[i]['total_prices'] = str(price) + 'Yuan'


# URL地址
@app.route('/thanks', methods=['POST'])
def thanks():
    return render_template('UI-4.html')


@app.route('/checkout', methods=['POST'])
def checkout():
    return render_template('UI-3.html', value=round(total_prices, 1))


@app.route('/detect', methods=['Get', 'POST'])
def detect():
    ret, frame = video_capture.read()

    path = 'static/images'
    cv2.imwrite(os.path.join(path, 'capture.jpg'), frame)

    det.loaddata()
    result = det.run()
    print(result)

    regu(result)
    print(regular_result)

    return render_template('UI-2.html', fanhui=regular_result)


@app.route('/')
def homepage():
    return render_template('homepage.html')


if __name__ == '__main__':
    app.run()
