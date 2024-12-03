# Scorebox Thresholding Program
# Contributors: Skylar Gallup <cwg7336@rit.edu>

import cv2
import sys
import numpy as np
from cv2.typing import MatLike


# Constants
LEFT_SIDE: str = "left"
RIGHT_SIDE: str = "right"
BOTH_SIDES: str = "both"
NO_SIDE: str = "none"

# Common CV pipeline tasks
def convert_to_grayscale(src: MatLike) -> MatLike:
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(src: MatLike) -> MatLike:
    return cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

def erode_dilate(src: MatLike, amount: int) -> MatLike:
    kernel = np.ones((amount, amount), np.uint8)
    eroded = cv2.erode(src, kernel, iterations = 1)
    dilated = cv2.dilate(eroded, kernel, iterations = 1)
    return dilated

def dilate(src: MatLike, amount: int) -> MatLike:
    kernel = np.ones((amount, amount), np.uint8)
    dilated = cv2.dilate(src, kernel, iterations = 1)
    return dilated

def get_centroid(contour: MatLike) -> tuple[int, int]:
    # Adding epsilon to the denominator moments to avoid divide-by-zero issues
    epsilon = sys.float_info.epsilon

    moments = cv2.moments(contour)
    centroid_x = int(moments['m10'] / (moments['m00'] + epsilon))
    centroid_y = int(moments['m01'] / (moments['m00'] + epsilon))
    return (centroid_x, centroid_y)
