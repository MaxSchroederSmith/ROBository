from motors import Motors
from time import time, sleep



mc = Motors()

 
# Move motor with the given ID at your set speed 
#mc.move_motor(motor_id,speed)      
#start_time = time() 

# Encoder board can be fragile - always use a try/except loop  
#while time() < start_time + run_time: 
#  mc.print_encoder_data() 
#  sleep(0.2)     # Use a sleep of at least 0.1, to avoid errors 
 
#mc.stop_motors()



def load_tray():
    
    start_time = time()
    
    #load card onto flipper
    mc.move_motor(2,-35)
    mc.move_motor(1,-45)
    
    while time() < start_time + 0.7:
        sleep(0.1)
    mc.stop_motor(1)
    
    while time() < start_time + 1.2:
        sleep(0.1)
    mc.stop_motor(2)
    
    #reverse motor 1 to reset deck
    start_time2 = time()
    mc.move_motor(1,35)
    while time() < start_time2 + 1: 
        sleep(0.2) 
        
    mc.stop_motor(1)
    
def flip_eject():
    
    #perform the flip
    mc.move_motor(0, 35)
    start_time = time()
    while time() < start_time + 1.8:
        sleep(0.1)

    mc.stop_motor(0)
    
    sleep(0.2)
    
    #mc.print_encoder_data() 
    
    eject()
    
    #reverse the flip
    start_time3 = time()
    mc.move_motor(0, -35)
    while time() < start_time3 + 1.8:
        sleep(0.1)

    mc.stop_motor(0)
    sleep(1)
    #mc.print_encoder_data()
    
def flip():
    #perform the flip
    mc.move_motor(0, 35)
    start_time = time()
    while time() < start_time + 1.8:
        sleep(0.1)

    mc.stop_motor(0)
    
    #reverse the flip
    start_time2 = time()
    mc.move_motor(0, -35)
    while time() < start_time2 + 1.8:
        sleep(0.1)

    mc.stop_motor(0)

def eject():
    #eject card
    start_time = time()
    mc.move_motor(2, -100)
    while time() < start_time + 1:
        sleep(0.1)
    
    mc.stop_motor(2)
    sleep(0.2)


def var_shoot_card(speed):
    start_time = time()
    mc.move_motor(1, speed)
    mc.move_motor(0,-50)
    
    while time() < start_time + 0.5:
        sleep(0.1)
    mc.stop_motor(0)
    
    while time() < start_time + 0.6:
        sleep(0.1)
    
    mc.stop_motor(1)
    start_time2 = time()
    mc.move_motor(0,35)
    while time() < start_time2 + 1.5: 
        sleep(0.2) 
        
    mc.stop_motor(0)

def shoot_card_no_flip():
    load_tray()
    eject()
    
    
def shoot_card_flip():
    load_tray()
    flip_eject()
    

    
def shoot_card2():
    mc.move_motor(0,-30)
    sleep(0.1)
    mc.move_motor(1, 100)
    sleep(0.9)
    mc.stop_motor(0)
    sleep(0.9)
    mc.stop_motor(1)
    sleep(0.1)
    mc.move_motor(0,35)
    sleep(0.8)
    mc.stop_motors()

#shoot_card_no_flip()
#mc = Motors()



#flip()

#eject()
#sleep(0.2)
#shoot_card_flip()

#for i in range(10):
#    shoot_card_no_flip()

#load_tray()

#flip_eject()

#shoot_card_no_flip()