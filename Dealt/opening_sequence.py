import stepper
from grove.grove_ws2813_rgb_led_strip import GroveWS2813RgbStrip
from rpi_ws281x import Color
import time
from time import sleep

PIN, COUNT = 18,60

strip = GroveWS2813RgbStrip(PIN, COUNT)


def allOff(strip):
    color = Color(0,0,0)
    for i in range(0, strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()

def wheel_of_colours(led_position):
    if led_position < 85:
        return Color(led_position*3, 255-led_position,0)
    elif led_position < 170:
        led_position -=85
        return Color(255-led_position*3,0, led_position*3)
    else:
        led_position-= 170
        return Color(0, led_position*3, 255-led_position*3)
    

def rainbowCycle(strip, wait_ms=20, iterations=10):
    for j in range(256*iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i,wheel_of_colours((int(i*256 / strip.numPixels()) + j) & 255))
        strip.show()
        time.sleep(wait_ms/1000.00)


rainbowCycle(strip)

allOff(strip)
                    
            
        

        




