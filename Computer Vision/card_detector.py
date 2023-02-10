#
# IMPORTS
#
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import use as mpl_use
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon

mpl_use('MacOSX')

#
# GLOBAL VARIABLES
#
imgs_dir = "pi_cards"  # Directory that will contain all the card images
RANK_DIFF_MAX = 5000
RANK_WIDTH = 70
RANK_HEIGHT = 125


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

    # Noise-reducing and edge-preserving filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge extraction
    edge = cv2.Canny(gray, 20, 200, L2gradient=True)

    # Find the contours in the edged image
    cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We suppose that the contour with largest area corresponds to the contour delimiting the card
    # MOST CENTERED CONTOUR???
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # Fill rank with solid colour
    img = cv2.fillPoly(img, pts=[cnt], color=(0, 0, 0))

    # Extract bounding box around rank
    x, y, w, h = cv2.boundingRect(cnt)
    extract = img[y:y + h, x:x + w]
    rotate = cv2.rotate(extract, cv2.ROTATE_90_CLOCKWISE)
    # Manipulating for better background/foreground contrast
    grayscale = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
    inverse = cv2.bitwise_not(grayscale)
    contrasted = modify_brightness_contrast(inverse, contrast=200)

    # todo Save the extracted image into the card_imgs file
    # DOESN'T KNOW WHAT THE SUIT IS YET THO????
    # FUNCTION LATER ON SORTING INTO CORRECT FOLDER??
    cv2.imwrite("extracted_suit.jpg", contrasted)

    # todo SORT OUT VALIDATION
    valid = True

    return valid, inverse


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


img_fn = imgs_dir + "/1_Ks.jpg"
img = cv2.imread(img_fn)  # VALIDATE THIS INCASE READING WRONG
img = crop_image(img)  # MAYBE WANT TO VALIDATE THIS AS WELL?
valid, extracted_rank = extract_rank(img, "test.jpg")
train_ranks = load_ranks('card_imgs/')
best_rank_match, rank_diff = match(extracted_rank, train_ranks)

print(best_rank_match)
