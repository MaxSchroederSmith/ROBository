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

def shoot_card():

    start_time = time()
    mc.move_motor(1,200)
    mc.move_motor(0,-50)
    
    while time() < start_time + 0.5: 
    	mc.print_encoder_data() 
    	sleep(0.2)     
    
    start_time2 = time()
    mc.move_motor(0,25)
    while time() < start_time2 + 2: 
    	mc.print_encoder_data() 
    	sleep(0.2) 
    	
    mc.stop_motors()

