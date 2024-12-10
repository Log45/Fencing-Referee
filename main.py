'''
Main program for processing videos and video streams
'''

import cv2
import cv2_common
from argparse import ArgumentParser, Namespace
from yolo_scorebox_classifier import ScoreboxDetectorClassifier
from cv2.typing import MatLike
from fencer_pose import FencerPoseClassifier

## Constants
SCOREBOX_MODEL_PATH = 'trained_models/scorebox_detect/scorebox_detect.pt'
FENCER_POSE_MODEL_PATH = 'trained_models/fencer_saber_keypoint/fencer_saber_keypoint.pt'

## Functions for Stream Handling

def get_stream(args: Namespace) -> cv2.VideoCapture:
    stream_method: str | None = args.mode
    while stream_method is None:
        stream_method = input('Select a stream method ("file", "webcam"): ')
        if stream_method not in ['file', 'webcam']:
            stream_method = None
    
    cap: cv2.VideoCapture
    match stream_method:
        case 'file':
            filename: str | None = args.filename
            while filename is None or filename == '':
                filename = input('Path to video file: ')
            cap = get_stream_file(filename)
        case 'webcam':
            cap = get_stream_webcam()
        case _:
            raise Exception('Invalid state reached')
    return cap
            

def get_stream_webcam(device_id: int = 0) -> cv2.VideoCapture:
    cap: cv2.VideoCapture = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise IOError(f'Could not open capture device with ID {device_id}')
    return cap

def get_stream_file(filename: str) -> cv2.VideoCapture:
    cap: cv2.VideoCapture = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f'Could not open video file {filename}')
    return cap

## Drawing Bounding Boxes and Keypoints
def draw_bounding_boxes(frame: MatLike, boxes: list, color=(0, 255, 0), thickness=3) -> MatLike:
    """
    Draw bounding boxes on a copy of the frame with a label and return the annotated frame.

    Args:
        frame (MatLike): The original image/frame.
        boxes (list): List of bounding boxes, where each box is [x_min, y_min, x_max, y_max].
        color (tuple): Color of the bounding box in BGR format. Default is green.
        thickness (int): Thickness of the bounding box lines. Default is 3.

    Returns:
        MatLike: A copy of the frame with bounding boxes and labels drawn.
    """
    annotated_frame = frame.copy()  # Create a copy of the original frame
    label = "scorebox"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    label_color = (0, 0, 0)  # Black text

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Draw the rectangle
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Determine label position
        label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        label_x = x_min
        label_y = y_min - 10 if y_min - 10 > 10 else y_min + 10  # Adjust if label goes out of frame
        
        # Draw label background for better visibility
        cv2.rectangle(
            annotated_frame,
            (label_x, label_y - label_size[1] - 2),
            (label_x + label_size[0] + 2, label_y + 2),
            color,
            thickness=cv2.FILLED,
        )
        
        # Draw the label text
        cv2.putText(
            annotated_frame,
            label,
            (label_x, label_y),
            font,
            font_scale,
            label_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated_frame


## Main Processing Loop

def main():
    # Parse command-line args
    parser = ArgumentParser(prog="Fencing-Referee", description="Automatically scores fencing bouts")
    parser.add_argument('-m', '--mode', choices=['webcam', 'file'], required=False)
    parser.add_argument('-f', '--filename', required=False)
    args = parser.parse_args()

    # Load scorebox model
    scorebox_detector = ScoreboxDetectorClassifier(SCOREBOX_MODEL_PATH)
    # Load fencer pose model
    fencer_pose_classifier = FencerPoseClassifier(FENCER_POSE_MODEL_PATH)

    # Open video stream
    cap: cv2.VideoCapture = get_stream(args)

    while cap.isOpened():
        # Get the next frame
        ret, frame = cap.read()
        if not ret:
            print('Stream ended; closing...')
            break

        # Process frame with scorebox detection
        scorebox_classification, scorebox_boxes = scorebox_detector.detect_and_classify(frame, debug=False)
        print("Scorebox Boxes:", scorebox_boxes)

        # Process frame with fencer pose classification
        labeled_frame_fencers, fencer_boxes, fencer_keypoints = fencer_pose_classifier.evaluate_on_input(frame)  # Process the same frame
        print("Fencer Boxes:", fencer_boxes)
        print("Fencer keypoints:", fencer_keypoints)

        # Result of the scorebox classification (left, right, both or none)
        print(f"Scorebox Classification: {scorebox_classification}")

        # Draw bounding boxes on the original frame for the scorebox
        annotated_frame = draw_bounding_boxes(frame, scorebox_boxes)
        # Resize the frame to reduce width and height by half
        frame_resized = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        # Display the frame and wait for 1/30th of a second
        cv2.imshow('stream', frame_resized)
        # cv2.waitKey(int(1000 / 30))
        if scorebox_classification != cv2_common.NO_SIDE:
            cv2.waitKey(0) # Pause when a valid classification is found
        else:
            cv2.waitKey(1) # Continue running

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
