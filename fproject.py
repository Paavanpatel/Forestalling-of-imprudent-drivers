import cv2
from iris import *
import time
import RPi.GPIO as gpio
from pi_detect_drowsiness import *
gpio.setmode(gpio.BOARD)
gpio.setup(5, gpio.OUT)
gpio.setup(7, gpio.IN, pull_up_down = gpio.PUD_UP)
eye_img = cv2.imread('/home/pi/CASIA/1/001_1_1.jpg')
print("Iris image recieved...")
if (recogn(eye_img) != -1):
    print("Welcome, you are Authenticated driver... ")
    gpio.output(5, gpio.HIGH)
    time.sleep(11)
    print(gpio.input(7))
    if gpio.input(7) == 1:
        print('Good, you are not drunk you can start the car :) ')
        drowsiness()
    else:
        print("Drunkness detected !!! you can't drive the car")
gpio.output(5, gpio.LOW)