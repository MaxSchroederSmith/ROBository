'''
    Stepper Motor interfacing with Raspberry Pi
    http:///www.electronicwings.com
'''
import RPi.GPIO as GPIO
from time import sleep
import sys

#assign GPIO pins for motor
motor_channel = (29,31,33,35)  
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
#for defining more than 1 GPIO channel as input/output use
GPIO.setup(motor_channel, GPIO.OUT)

defaultSpeed = 0.02




def forward(jumps):
    accel('c')
    for i in range((jumps-1)*20):
        clockwise(defaultSpeed)
    decel('c')


def backward(jumps):
    accel('a')
    for i in range((jumps-1)*20):
        anticlockwise(defaultSpeed)
    decel('a')
    
    
        
def accel(direction):
    sleepTime = 0.1
    for i in range(5):
        for j in range(3):
            if direction == "c":
                clockwise(sleepTime)
            elif direction == "a":
                anticlockwise(sleepTime)
        sleepTime -= 0.02


def decel(direction):
    sleepTime = 0.02
    for i in range(5):
        for j in range(2):
            if direction == "c":
                clockwise(sleepTime)
            elif direction == "a":
                anticlockwise(sleepTime)
        sleepTime += 0.02
            
            
def clockwise(sleepTime):
    GPIO.output(motor_channel, (GPIO.HIGH,GPIO.LOW,GPIO.HIGH,GPIO.LOW))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.LOW,GPIO.HIGH,GPIO.HIGH,GPIO.LOW))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.LOW,GPIO.HIGH,GPIO.LOW,GPIO.HIGH))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.HIGH,GPIO.LOW,GPIO.LOW,GPIO.HIGH))
    sleep(sleepTime)
    

    
def anticlockwise(sleepTime):
    GPIO.output(motor_channel, (GPIO.HIGH,GPIO.LOW,GPIO.LOW,GPIO.HIGH))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.LOW,GPIO.HIGH,GPIO.LOW,GPIO.HIGH))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.LOW,GPIO.HIGH,GPIO.HIGH,GPIO.LOW))
    sleep(sleepTime)
    GPIO.output(motor_channel, (GPIO.HIGH,GPIO.LOW,GPIO.HIGH,GPIO.LOW))
    sleep(sleepTime)
    
    

forward(20)

            