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
imgs_dir = "example_cards"  # Directory that will contain all the card images
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

class Train_ranks:
    """Structure to store information about train rank images."""
    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks


def crop_image(image):
    y = image.shape[0]
    x = image.shape[1]

    y1 = int(y / 3) - 20
    y2 = int(2 * y / 3) + 20
    x1 = int(x / 3) - 20
    x2 = int(2 * x / 3) + 20

    return image[y1:y2, x1:x2]


def varianceOfLaplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def extract_rank(img, output_fn=None, min_focus=20, debug=False):
    imgwarp = None

    # Check the image is not too blurry
    focus = varianceOfLaplacian(img)
    if focus < min_focus:
        if debug: print("Focus too low :", focus)
        return False, None

    # Convert in gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image("2grayscaled", gray)
    
    #gray = modify_brightness_contrast(gray, brightness=350, contrast=200)

    # Noise-reducing and edge-preserving filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    save_image("3noise-reduced", gray)

    # Edge extraction
    edge = cv2.Canny(gray, 20, 200)
    save_image("4edge_extracted", edge)

    # Find the contours in the edged image
    _, cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # if cnts empty not valid

    # We suppose that the 2 contours with largest area corresponds to the contour delimiting the card
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0:2]
    minimum_x = 1000000000000
    for contour in cnts:
        if cv2.boundingRect(contour)[0] < minimum_x:
            # Out of the 2 largest contours, one on the left will be the rank
            cnt = contour
            minimum_x = cv2.boundingRect(contour)[1]

    # Fill rank with solid colour
    img = cv2.fillPoly(img, pts=[cnt], color=(0, 0, 0))
    save_image("5rank_filled", img)

    # Extract bounding box around rank
    x, y, w, h = cv2.boundingRect(cnt)
    extract = img[y:y + h, x:x + w]
    rotate = cv2.rotate(extract, cv2.ROTATE_90_CLOCKWISE)
    save_image("6rank_extracted", rotate)
    # Manipulating for better background/foreground contrast
    grayscale = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
    inverse = cv2.bitwise_not(grayscale)
    contrasted = modify_brightness_contrast(inverse, contrast=200)
    save_image("7rank_extracted_manipulated", contrasted)

    # todo SORT OUT VALIDATION
    valid = True

    return valid, contrasted


def modify_brightness_contrast(img, brightness=255,
                               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    return cal


def match(extracted, train_ranks):
    best_rank_match_diff = 10000
    best_rank_match_name = "Unknown"
    extracted = cv2.resize(extracted, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
    i = 0
    for Trank in train_ranks:
        diff_img = cv2.absdiff(extracted, Trank.img)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_diff_img = diff_img
            best_rank_match_diff = rank_diff
            best_rank_name = Trank.name

    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name
    return best_rank_match_name, best_rank_match_diff

def display(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(50)

def save_image(name, img):
    cv2.imwrite(output_dir + "/" + name + ".jpg", img)

def main(debug=False):
    #img_fn = imgs_dir + "/1_Ks.jpg"
    #img = cv2.imread(img_fn)  # VALIDATE THIS INCASE READING WRONG

    if not debug:
        #camera.start_preview()
        if lights:
            allOn(strip)
        init_camera()
        capture_dir = output_dir+'/pi_card.jpg'
        camera.capture(capture_dir)
        sleep(1)
        camera.close()
        #camera.stop_preview()
        if lights:
            allOff(strip)
        img = cv2.imread(capture_dir)
        
    #img = cv2.imread("output/pi_card.jpg")
    img = crop_image(img)  # MAYBE WANT TO VALIDATE THIS AS WELL?
    save_image("1cropped",img)
    valid, extracted_rank = extract_rank(img, "test.jpg")
    # todo Save the extracted image into the card_imgs file
    # DOESN'T KNOW WHAT THE SUIT IS YET THO????
    # FUNCTION LATER ON SORTING INTO CORRECT FOLDER??

    train_ranks = load_ranks(rank_dataset_dir)
    best_rank_match, rank_diff = match(extracted_rank, train_ranks)
    
    return best_rank_match
#print(main())