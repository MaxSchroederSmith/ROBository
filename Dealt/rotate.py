from motors import Motors
from time import time, sleep



mc = Motors()

def rotate_axle(player):
    for i in range(player):
        start_time = time()
        mc.move_motor(3,-60)
        while time() < start_time + 1:
            sleep(0.1)
        mc.stop_motor(3)
        sleep(2)
    start_time = time()
    mc.move_motor(3,60)
    while time() < start_time + player:
        sleep(0.1)
    mc.stop_motor(3)
    for i in range(player):
        start_time = time()
        mc.move_motor(3,60)
        while time() < start_time + 1:
            sleep(0.1)
        mc.stop_motor(3)
        sleep(2)
rotate_axle(2)
    