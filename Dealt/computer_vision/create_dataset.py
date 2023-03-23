#!python3

#
# IMPORTS
#
from grove.grove_ws2813_rgb_led_strip import GroveWS2813RgbStrip
from rpi_ws281x import Color
from picamera import PiCamera
import time
from time import sleep
import numpy as np
import cv2

#
# GLOBAL VARIABLES
#
#camera = PiCamera()
dataset_dir = "rank_dataset"  # Directory that will contain all the card images
output_dir = "output"
rank_dataset_dir = "rank_dataset/"

RANK_DIFF_MAX = 5000
RANK_WIDTH = 70
RANK_HEIGHT = 125

lights = True
PIN, COUNT = 18,60
try:
    strip = GroveWS2813RgbStrip(PIN, COUNT)
except:
    lights = False

def init_camera():
    global camera
    camera = PiCamera()
    
def allOn(strip, wait_ms=10000):
    color = Color(255,255,255)
    for i in range(0, strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        
def allOff(strip):
    color = Color(0,0,0)
    for i in range(0, strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()