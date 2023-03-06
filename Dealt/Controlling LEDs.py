from gpiozero import LED 
import time

red = LED(17) 
 
while True: 
    red.on()    #turn led on
    time.sleep(1)
    red.on()    #turn led on
    time.sleep(1)