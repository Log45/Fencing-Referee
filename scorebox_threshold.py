# Scorebox Thresholding Program
# Contributors: Skylar Gallup <cwg7336@rit.edu>

import cv2
import sys
import numpy as np
from cv2.typing import MatLike
from typing import Sequence


LEFT_SIDE: str = "left"
RIGHT_SIDE: str = "right"
BOTH_SIDES: str = "both"
NO_SIDE: str = "none"


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

def threshold_red(src: MatLike) -> MatLike:
    # return cv2.threshold(src, 127.0, 255.0, cv2.THRESH_BINARY)[1]
    #return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -10)
    # Take main red threshold, including wrapping around the zero side of the hue range
    hue_range = 15
    low_range = cv2.inRange(src, (0, 30, 150), (hue_range, 255, 255))
    high_range = cv2.inRange(src, (179 - hue_range, 30, 150), (179, 255, 255))
    main_threshold = cv2.bitwise_or(low_range, high_range)

    # Threshold orange text and subtract to remove text
    text_threshold = cv2.inRange(src, (0, 50, 150), (30, 150, 255))
    text_threshold = dilate(text_threshold, 10)
    return cv2.bitwise_and(main_threshold, cv2.bitwise_not(text_threshold))

def threshold_green(src: MatLike) -> MatLike:
    # return cv2.threshold(src, 127.0, 255.0, cv2.THRESH_BINARY)[1]
    #return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -10)
    hue_range = 15
    return cv2.inRange(src, (60 - hue_range, 0, 150), (60 + hue_range, 255, 255))

def find_edges(src: MatLike) -> MatLike:
    return cv2.Canny(src, 50, 100)

def find_contours(src: MatLike) -> Sequence[MatLike]:
    contours = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    hulls: Sequence[MatLike] = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        hulls.append(hull)
    return hulls

def draw_contours(src: MatLike, contours: Sequence[MatLike], color: tuple[int, int, int]) -> MatLike:
    if len(contours) > 0:
        return cv2.drawContours(src, contours, -1, color, 2)
    else:
        return src

def filter_contours(contours: Sequence[MatLike], side: str, height: int, width: int) -> Sequence[MatLike]:
    # Input validation
    if side not in [LEFT_SIDE, RIGHT_SIDE]:
        raise TypeError('"side" must be "left" or "right"')

    # Only take contours on the correct "side" of the image
    if side == LEFT_SIDE:
        contours = list(filter(lambda cnt: get_centroid(cnt)[0] < (width * 0.33), contours))
    elif side == RIGHT_SIDE:
        contours = list(filter(lambda cnt: get_centroid(cnt)[0] > (width * 0.67), contours))

    # # Take the "side"-most contour
    # if len(contours) > 0:
    #     if side == LEFT_SIDE:
    #         return [min(contours, key = lambda cnt: get_centroid(cnt)[0])]
    #     elif side == RIGHT_SIDE:
    #         return [max(contours, key = lambda cnt: get_centroid(cnt)[0])]
    #     else:
    #         return contours
    # else:
    #     return contours

    # Take the largest contour
    if len(contours) > 0:
        return [max(contours, key = lambda cnt: cv2.contourArea(cnt))]
    else:
        return contours

def get_centroid(contour: MatLike) -> tuple[int, int]:
    # Adding epsilon to the denominator moments to avoid divide-by-zero issues
    epsilon = sys.float_info.epsilon

    moments = cv2.moments(contour)
    centroid_x = int(moments['m10'] / (moments['m00'] + epsilon))
    centroid_y = int(moments['m01'] / (moments['m00'] + epsilon))
    return (centroid_x, centroid_y)

def pipeline(src: MatLike) -> str:
    height, width = src.shape[:2]

    hsv_img = convert_to_hsv(src)
        
    green_threshold_img = threshold_green(hsv_img)
    green_threshold_img = erode_dilate(green_threshold_img, 5)

    green_contours = find_contours(green_threshold_img)
    green_contours = filter_contours(green_contours, RIGHT_SIDE, height, width)
    green_contours_img = draw_contours(src, green_contours, (0, 255, 0))

    red_threshold_img = threshold_red(hsv_img)
    red_threshold_img = erode_dilate(red_threshold_img, 5)

    red_contours = find_contours(red_threshold_img)
    red_contours = filter_contours(red_contours, LEFT_SIDE, height, width)
    contours_img = draw_contours(green_contours_img, red_contours, (0, 0, 255))

    cv2.imshow("src", src)
    # cv2.imshow("threshold_img", threshold_img)
    cv2.imshow("contours_img", contours_img)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)

    if len(red_contours) > 0 and len(green_contours) > 0:
        return BOTH_SIDES
    elif len(red_contours) > 0 and len(green_contours) == 0:
        return LEFT_SIDE
    elif len(red_contours) == 0 and len(green_contours) > 0:
        return RIGHT_SIDE
    else:
        return NO_SIDE


if __name__ == "__main__":
    filename = input("Relative location of image to process: ")
    
    filename = "left-1.png"
    img = cv2.imread(filename)
    output = pipeline(img)
    print(f"Output: {output}")

    # filenames = ["left-1.png", "right-1.png", "none-1.png", "both-1.png", "both-2.png"]
    # for fname in filenames:
    #     img = cv2.imread(fname)
    #     pipeline(img)
    #     cv2.waitKey(0)
