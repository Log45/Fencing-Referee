# Scorebox Thresholding Program
# Contributors: Skylar Gallup <cwg7336@rit.edu>

import cv2
from cv2.typing import MatLike


LEFT_OUTPUT: str = "left"
RIGHT_OUTPUT: str = "right"
BOTH_OUTPUT: str = "both"
NONE_OUTPUT: str = "none"


def convert_to_grayscale(src: MatLike) -> MatLike:
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(src: MatLike) -> MatLike:
    return cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

def threshold_red(src: MatLike) -> MatLike:
    # return cv2.threshold(src, 127.0, 255.0, cv2.THRESH_BINARY)[1]
    #return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -10)
    hue_range = 15
    low_range = cv2.inRange(src, (0, 0, 40), (hue_range, 255, 255))
    high_range = cv2.inRange(src, (179 - hue_range, 0, 40), (179, 255, 255))
    return cv2.bitwise_or(low_range, high_range)

def threshold_green(src: MatLike) -> MatLike:
    # return cv2.threshold(src, 127.0, 255.0, cv2.THRESH_BINARY)[1]
    #return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, -10)
    hue_range = 15
    return cv2.inRange(src, (60 - hue_range, 0, 40), (60 + hue_range, 255, 255))

def find_edges(src: MatLike) -> MatLike:
    return cv2.Canny(src, 50, 50)

def pipeline(src: MatLike) -> str:
    grayscale_img = convert_to_grayscale(src)
    hsv_img = convert_to_hsv(src)
        
    threshold_img = threshold_green(hsv_img)
    edges = find_edges(grayscale_img)

    cv2.imshow("src", src)
    cv2.imshow("threshold_img", threshold_img)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)

    return NONE_OUTPUT


if __name__ == "__main__":
    #filename = input("Relative location of image to process: ")
    
    # filename = "left-1.png"
    # img = cv2.imread(filename)
    # output = pipeline(img)
    # print(f"Output: {output}")

    filenames = ["left-1.png", "right-1.png", "none-1.png", "both-1.png", "both-2.png"]
    for fname in filenames:
        img = cv2.imread(fname)
        pipeline(img)
        cv2.waitKey(0)
