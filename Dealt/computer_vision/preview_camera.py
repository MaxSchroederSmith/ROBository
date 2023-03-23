from picamera import PiCamera
camera = PiCamera()
camera.start_preview(fullscreen=False, window=(100,20,640,480))
input()
camera.close()
'''
try:
    
    pass
finally:
    camera.close()
'''