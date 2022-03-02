import cv2
from PIL import Image

WIDTH = 6
HEIGHT = 6


def createImage(width, height):
    img = Image.new(mode="RGB", size=(width, height), color=(209, 123, 193))
    img.save("../test_image.PNG")
    return 0


def readImage(filename):
    img = cv2.imread(filename)
    return img


createImage(WIDTH, HEIGHT)
