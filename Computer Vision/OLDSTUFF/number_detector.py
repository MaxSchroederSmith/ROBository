import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

BKG_THRESH = 10
CARD_THRESH = 30


class QueryCard:
    def __init__(self):
        #self.contour = []  # Contour of card
        #self.width, self.height = 0, 0  # Width and height of card
        #self.corner_pts = []  # Corner points of card
        #self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = []  # Thresholded, sized image of card's rank
        #self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        #self.best_suit_match = "Unknown"  # Best matched suit
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        #self.suit_diff = 0  # Difference between suit image and best matched train suit image


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


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_w, img_h = np.shape(image)[:2]
    print(img_w)
    print(img_h)
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh

train_ranks = load_ranks('card_imgs/')
image_filename = 'card2'
image_path = 'example_cards/' + image_filename + '.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = preprocess_image(img)
plt.imshow(img)
plt.show()
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(img, contours, -1, (0,255,0), 3)


qCard = QueryCard()
qCard.contour = contours
