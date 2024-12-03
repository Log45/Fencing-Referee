# Scorebox Thresholding Program
# Contributors: Skylar Gallup <cwg7336@rit.edu>

import cv2
from cv2_common import *
from cv2.typing import MatLike
from typing import Sequence


class ScoreboxThresholdClassifier:
    def threshold_red(self, src: MatLike) -> MatLike:
        # Take main red threshold, including wrapping around the zero side of the hue range
        hue_range = 15
        low_range = cv2.inRange(src, (0, 30, 150), (hue_range, 255, 255))
        high_range = cv2.inRange(src, (179 - hue_range, 30, 150), (179, 255, 255))
        main_threshold = cv2.bitwise_or(low_range, high_range)

        # Threshold orange text and subtract to remove text
        text_threshold = cv2.inRange(src, (0, 50, 150), (30, 150, 255))
        text_threshold = dilate(text_threshold, 10)
        return cv2.bitwise_and(main_threshold, cv2.bitwise_not(text_threshold))

    def threshold_green(self, src: MatLike) -> MatLike:
        hue_range = 15
        return cv2.inRange(src, (60 - hue_range, 0, 150), (60 + hue_range, 255, 255))

    def find_contours(self, src: MatLike) -> Sequence[MatLike]:
        contours = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        hulls: Sequence[MatLike] = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            hulls.append(hull)
        return hulls

    def draw_contours(self, src: MatLike, contours: Sequence[MatLike], color: tuple[int, int, int]) -> MatLike:
        if len(contours) > 0:
            return cv2.drawContours(src, contours, -1, color, 2)
        else:
            return src

    def filter_contours(self, contours: Sequence[MatLike], side: str, height: int, width: int) -> Sequence[MatLike]:
        # Input validation
        if side not in [LEFT_SIDE, RIGHT_SIDE]:
            raise TypeError('"side" must be "left" or "right"')

        # Only take contours on the correct "side" of the image
        if side == LEFT_SIDE:
            contours = list(filter(lambda cnt: get_centroid(cnt)[0] < (width * 0.33), contours))
        elif side == RIGHT_SIDE:
            contours = list(filter(lambda cnt: get_centroid(cnt)[0] > (width * 0.67), contours))

        # Take the largest contour
        if len(contours) > 0:
            return [max(contours, key = lambda cnt: cv2.contourArea(cnt))]
        else:
            return contours

    def classify(self, src: MatLike, show_images: bool = False) -> str:
        height, width = src.shape[:2]

        hsv_img = convert_to_hsv(src)
            
        green_threshold_img = self.threshold_green(hsv_img)
        green_threshold_img = erode_dilate(green_threshold_img, 5)

        green_contours = self.find_contours(green_threshold_img)
        green_contours = self.filter_contours(green_contours, RIGHT_SIDE, height, width)

        red_threshold_img = self.threshold_red(hsv_img)
        red_threshold_img = erode_dilate(red_threshold_img, 5)

        red_contours = self.find_contours(red_threshold_img)
        red_contours = self.filter_contours(red_contours, LEFT_SIDE, height, width)

        if show_images:
            green_contours_img = self.draw_contours(src, green_contours, (0, 255, 0))
            contours_img = self.draw_contours(green_contours_img, red_contours, (0, 0, 255))
            cv2.imshow("contours_img", contours_img)
            cv2.waitKey(0)

        if len(red_contours) > 0 and len(green_contours) > 0:
            return BOTH_SIDES
        elif len(red_contours) > 0 and len(green_contours) == 0:
            return LEFT_SIDE
        elif len(red_contours) == 0 and len(green_contours) > 0:
            return RIGHT_SIDE
        else:
            return NO_SIDE

    def classify_file(self, filename: str, show_images: bool = False) -> str:
        src = cv2.imread(filename)
        return self.classify(src, show_images)


if __name__ == "__main__":
    classifier = ScoreboxThresholdClassifier()
    filename = input("Relative location of image to process: ")
    if filename == "all":
        filenames = ["left-1.png", "right-1.png", "none-1.png", "both-1.png", "both-2.png"]
        for filename in filenames:
            output = classifier.classify_file(filename, True)
            print(f"Output of {filename}: {output}")
    else:
        output = classifier.classify_file(filename, True)
        print(f"Output: {output}")
