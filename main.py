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
FENCER_POSE_MODEL_PATH = 'trained_models/fencer_keypoint/fencer_keypoint.pt'

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
def annotate_boxes(annotated_frame, boxes, color, thickness, label_prefix):
    """Helper function to annotate boxes with labels."""
    for box, confidence in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        # Draw the bounding box
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, thickness)
        # Create label with confidence
        label = f"{label_prefix}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        label_y_min = max(y_min - 10, 0)  # Ensure label doesn't go off-frame
        cv2.rectangle(
            annotated_frame, 
            (x_min, label_y_min - label_size[1] - baseline),
            (x_min + label_size[0], label_y_min + baseline),
            color, 
            thickness=cv2.FILLED
        )
        cv2.putText(
            annotated_frame, 
            label, 
            (x_min, label_y_min), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 0),  # Black text
            2
        )

def annotate_keypoints(frame: MatLike, 
                       keypoints: list, 
                       color=(0, 255, 0), 
                       radius=5, 
                       thickness=3, 
                       label_prefix=None) -> MatLike:
    """
    Annotate keypoints on a frame.

    Args:
        frame (MatLike): The original image/frame.
        keypoints (list): List of keypoints, where each entry is a tuple of (x, y).
        color (tuple): Color of the keypoints in BGR format. Default is green.
        radius (int): Radius of the keypoint circles. Default is 5.
        thickness (int): Thickness of the circle edge. Default is 2.
        label_prefix (str): Optional prefix for labels, e.g., "Left Fencer" or "Right Fencer".

    Returns:
        MatLike: A copy of the frame with annotated keypoints.
    """
    for keypoint in keypoints:
        for xy in keypoint:
            # Extract coordinates and confidence if available
            x, y = xy
            x, y = int(x), int(y)  # Convert to integers for OpenCV

            # Draw the keypoint circle
            cv2.circle(frame, (x, y), radius, color, thickness)

            # Add confidence as a label if available
            '''
            if confidence is not None:
                label = f"{label_prefix}: {confidence:.2f}" if label_prefix else f"{confidence:.2f}"
                cv2.putText(annotated_frame, label, (x + radius, y - radius), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            '''

def annotate_frame_with_boxes(frame: MatLike, 
                              scorebox_boxes: list, 
                              fencer_boxes_left: list,
                              fencer_boxes_right: list, 
                              fencer_keypoints_left: list,
                              fencer_keypoints_right: list,
                              left_movement: float,
                              right_movement: float,
                              scorebox_color=(200, 255, 200), 
                              fencer_color=(255, 200, 200), 
                              thickness=2) -> MatLike:
    """
    Annotate a frame with bounding boxes for both scoreboxes and fencers, 
    as well as the left and right movement values displayed in boxes at the bottom corners.

    Args:
        frame (MatLike): The original image/frame.
        scorebox_boxes (list): List of scorebox bounding boxes and confidence scores, 
                               where each entry is [[x_min, y_min, x_max, y_max], confidence].
        fencer_boxes_left (list): List of left fencer bounding boxes and confidence scores.
        fencer_boxes_right (list): List of right fencer bounding boxes and confidence scores.
        left_movement (float): The movement of the left fencer in pixels.
        right_movement (float): The movement of the right fencer in pixels.
        scorebox_color (tuple): Color of the scorebox bounding boxes in BGR format. Default is green.
        fencer_color (tuple): Color of the fencer bounding boxes in BGR format. Default is blue.
        thickness (int): Thickness of the bounding box lines. Default is 3.

    Returns:
        MatLike: A copy of the frame with annotated bounding boxes and movement info.
    """
    annotated_frame = frame.copy()  # Create a copy of the original frame

    # Annotate scoreboxes
    annotate_boxes(annotated_frame, scorebox_boxes, scorebox_color, thickness, "Scorebox")

    # Annotate fencers
    annotate_boxes(annotated_frame, fencer_boxes_left, (255, 200, 200), thickness, "Left Fencer")
    annotate_boxes(annotated_frame, fencer_boxes_right, (200, 200, 255), thickness, "Right Fencer")

    # Annotate keypoints
    annotate_keypoints(annotated_frame, fencer_keypoints_left, 
                                     color=(255, 100, 100), label_prefix="Left")
    annotate_keypoints(annotated_frame, fencer_keypoints_right, 
                                     color=(100, 100, 255), label_prefix="Right")

    # Add left and right movement text at the bottom-left and bottom-right corners
    height, width = annotated_frame.shape[:2]  # Get the frame dimensions

    # Left movement display (bottom-left corner)
    left_movement_text = f"Left Movement: {left_movement:.2f} px"
    cv2.putText(annotated_frame, left_movement_text, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Right movement display (bottom-right corner)
    right_movement_text = f"Right Movement: {right_movement:.2f} px"
    text_size = cv2.getTextSize(right_movement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    right_text_x = width - text_size[0] - 10  # Make sure text is within the frame
    cv2.putText(annotated_frame, right_movement_text, (right_text_x, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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
        (fencer_boxes_left, fencer_keypoints_left, fencer_boxes_right, fencer_keypoints_right, 
         left_movement, right_movement) = fencer_pose_classifier.evaluate_on_input(frame)  # Process the same frame
        #print("Left Fencer Boxes:", fencer_boxes_left)
        #print("Left Fencer keypoints:", fencer_keypoints_left, "\n")
        #print("Right Fencer Boxes:", fencer_boxes_right)
        #print("Right Fencer keypoints:", fencer_keypoints_right, "\n")
        #print("Left Fencer movement", left_movement)
        #print("Right Fencer movement", right_movement)

        # Result of the scorebox classification (left, right, both or none)
        #print(f"Scorebox Classification: {scorebox_classification}")

        # Draw bounding boxes on the original frame for the scorebox
        annotated_frame = annotate_frame_with_boxes(frame, scorebox_boxes, fencer_boxes_left, fencer_boxes_right, 
                                                    fencer_keypoints_left, fencer_keypoints_right, left_movement, right_movement)
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
