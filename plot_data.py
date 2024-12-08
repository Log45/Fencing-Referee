'''
Helper file for visualizing labels and output data.
'''

import cv2
from cv2.typing import MatLike

def draw_rectangle(src: MatLike, data: tuple[float, float, float, float]) -> MatLike:
    img_width, img_height = src.shape[:2]

    center_x = data[0] * img_width
    center_y = data[1] * img_height
    extent_x = data[2] * img_width
    extent_y = data[3] * img_height

    x1 = int(center_x - (extent_x / 2))
    y1 = int(center_y - (extent_y / 2))
    x2 = int(center_x + (extent_x / 2))
    y2 = int(center_y + (extent_y / 2))

    return cv2.rectangle(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

def main():
    # Load image
    # filename: str = input('Path to file: ')
    filename: str = 'datasets/test/images/0_76_jpg.rf.c09434ad16cffe8c89de01a80e288b83.jpg'
    img: MatLike = cv2.imread(filename)

    # Scale to 640x640
    scaled_img = cv2.resize(img, (640, 640))

    # Draw bounding box
    scaled_img = draw_rectangle(scaled_img, (0.46328125, 0.38671875, 0.13359375, 0.05859375))
    scaled_img = draw_rectangle(scaled_img, (0.4609375, 0.71953125, 0.04453125, 0.0421875))

    # Show result
    cv2.imshow("Annotated image", scaled_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
