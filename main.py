'''
Main program for processing videos and video streams
'''

import cv2
import cv2_common
from argparse import ArgumentParser, Namespace
from yolo_scorebox_classifier import ScoreboxDetectorClassifier
from cv2.typing import MatLike
from fencer_pose import FencerPoseClassifier
from nn_pose_classifier import SimpleNNClassifier
import torch
import numpy as np

## Constants
SCOREBOX_MODEL_PATH = 'trained_models/scorebox_detect/scorebox_detect.pt'
FENCER_POSE_MODEL_PATH = 'trained_models/fencer_keypoint/fencer_keypoint.pt'
POSE_CLASSIFIER_MODEL_PATH = 'trained_models/pose_classifier/pose_classifier.pth'
POINT_DECISION_TREE_PATH = None # 'trained_models/point_decider/point_decider.pkl'

NO_POINT = 0
POINT_LEFT = 1
POINT_RIGHT = 2

POINT_DICT = {
    NO_POINT: "No Point",
    POINT_LEFT: "Left Fencer",
    POINT_RIGHT: "Right Fencer"
}

POSES = ["En Garde", "Lunge", "Parry", "None"]

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
            1
        )

def annotate_frame_with_boxes(frame: MatLike, 
                              scorebox_boxes: list, 
                              fencer_boxes_left: list,
                              fencer_boxes_right: list, 
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
    annotate_boxes(annotated_frame, fencer_boxes_left, fencer_color, thickness, "Left Fencer")
    annotate_boxes(annotated_frame, fencer_boxes_right, fencer_color, thickness, "Right Fencer")

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

def interim_point_decider(left_pose, right_pose, left_movement, right_movement):
    # Logic Table for determining the point (preferably will be changed to a DecisionTree in the future)
    
    # First check if one of the fencers is retreating
    if left_movement < -10:
        return POINT_RIGHT
    elif right_movement > 10:
        return POINT_LEFT
    
    if left_pose == right_pose:
        if left_movement > abs(right_movement):
            return POINT_LEFT
        elif abs(right_movement) > left_movement:
            return POINT_RIGHT          
    elif left_pose == 1 and right_pose == 0:
        return POINT_LEFT
    elif left_pose == 0 and right_pose == 1:
        return POINT_RIGHT

    return NO_POINT

def normalize_keypoints_to_bbox(img, keypoints, bbox, imgsize=(640, 640)):
    """
    Normalize keypoints to a bounding box.
    
    Parameters:
        keypoints (list of tuples): List of keypoints [(x, y), ...], un-normalized.
        bbox (tuple): Bounding box in (x1 y1 x2 y2), un-normalized.
    
    Returns:
        list of tuples: Keypoints normalized to the bounding box.
    """
    h, w, _ = img.shape
    
    x1, y1, x2, y2 = bbox
    
    x_min = x1 / w
    x_max = x2 / w
    y_min = y1 / h
    y_max = y2 / h
    
    width = x_max - x_min
    height = y_max - y_min
    
    normalized_keypoints = []

    for x, y in keypoints:
        # print("Bbox:", x_min, y_min, width, height)
        x_normalized = ((x/640)-x_min) / width
        y_normalized = ((y/640)-y_min) / height
        normalized_keypoints.extend([x_normalized, y_normalized])
    
    return normalized_keypoints
    

def determine_point(frame, cap, pose_classifier: SimpleNNClassifier, point_decider, scorebox_classification, fencer_boxes_left, fencer_boxes_right, left_keypoints, right_keypoints, left_movement, right_movement):
    # Logic Table for determining the point
    if scorebox_classification == cv2_common.LEFT_SIDE:
        return POINT_LEFT
    elif scorebox_classification == cv2_common.RIGHT_SIDE:
        return POINT_RIGHT
    else:
        # print("Left Keypoints:", left_keypoints)
        # print("Right Keypoints:", right_keypoints)
        # print("Left Fencer Boxes:", fencer_boxes_left)
        # print("Right Fencer Boxes:", fencer_boxes_right)
        left_keypoints = normalize_keypoints_to_bbox(frame, left_keypoints[0], fencer_boxes_left[0][0])
        right_keypoints = normalize_keypoints_to_bbox(frame, right_keypoints[0], fencer_boxes_right[0][0])
        # print("Normalized Left Keypoints:", left_keypoints)
        # print("Normalized Right Keypoints:", right_keypoints)
        left_pose_probs = pose_classifier.predict_probs(torch.as_tensor(left_keypoints, device=pose_classifier.device).unsqueeze(0))
        right_pose_probs = pose_classifier.predict_probs(torch.as_tensor(right_keypoints, device=pose_classifier.device).unsqueeze(0))
        
        print("Left Pose Max: ", torch.max(left_pose_probs).item(), "Left Prediction: ", POSES[torch.argmax(left_pose_probs)],"\tRight Pose Max: ", torch.max(right_pose_probs).item(), "Right Prediction: ", POSES[torch.argmax(right_pose_probs)])
        
        if torch.max(left_pose_probs) > 0.45 and torch.max(right_pose_probs) > 0.45:
            # Only make a decision if the model is confident
            left_pose = torch.argmax(left_pose_probs)
            right_pose = torch.argmax(right_pose_probs)  
            return point_decider(left_pose, right_pose, left_movement, right_movement)
    return NO_POINT

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
    # Load pose classifier model
    pose_classifier = torch.load(POSE_CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu'))
    pose_classifier.to(pose_classifier.device)
    # Load point decision tree model
    # point_decider = joblib.load(POINT_DECISION_TREE_PATH)
    

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
        annotated_frame = annotate_frame_with_boxes(frame, scorebox_boxes, fencer_boxes_left, fencer_boxes_right, left_movement, right_movement)
        # Resize the frame to reduce width and height by half
        frame_resized = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        # Display the frame and wait for 1/30th of a second
        cv2.imshow('stream', frame_resized)
        # cv2.waitKey(int(1000 / 30))
        if scorebox_classification != cv2_common.NO_SIDE:
            print(POINT_DICT[determine_point(frame, cap, pose_classifier, interim_point_decider, scorebox_classification, fencer_boxes_left, fencer_boxes_right, fencer_keypoints_left, fencer_keypoints_right, left_movement, right_movement)])
            cv2.waitKey(1) # Pause when a valid classification is found
        else:
            cv2.waitKey(1) # Continue running

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
