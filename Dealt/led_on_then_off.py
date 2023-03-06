from grove.grove_ws2813_rgb_led_strip import GroveWS2813RgbStrip
from rpi_ws281x import Color
import time, math

PIN, COUNT = 18,60
strip = GroveWS2813RgbStrip(PIN, COUNT)

def allOn(strip, wait_ms=10000):
    color = Color(255,255,255)
    for i in range(0, strip.numPixels()):
        strip.setPixelColor(i, Color(aya
                                     ,255,255))
        strip.show()
    time.sleep(wait_ms/1000.0)
     
def colourOn(strip, wait_ms=10000):
    color = Color(255,255,255)
    for number in range(200):
        for i in range(0, strip.numPixels()):
            red = int(((math.sin(number+i*(math.pi/4))+1)*(255/2)))
            green = int(((math.sin(number+i*(math.pi/4)+((math.pi*2)/3))+1)*(255/2)))
            blue = int(((math.sin(number+i*(math.pi/4)+((math.pi*4)/3))+1)*(255/2)))
            strip.setPixelColor(i, Color(red
                                        ,green,blue))
            strip.show()
    time.sleep(wait_ms/1000.0)
     
def allOff(strip):
    color = Color(0,0,0)
    for i in range(0, strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        
colourOn(strip)
allOff(strip)