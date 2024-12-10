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

def draw_keypoints(frame: MatLike, keypoints: list[list[tuple[float, float]]], color=(0, 0, 255), keypoint_radius=5, bone_thickness=2) -> MatLike:
    # Define a graph of nodes (keypoints) and edges (bones) to draw
    nodes: dict[str, int] = {
        'head': 0,
        'left-elbow': 1,
        'left-foot': 2,
        'left-hip': 3,
        'left-knee': 4,
        'left-shoulder': 5,
        'left-wrist': 6,
        'right-elbow': 7,
        'right-foot': 8,
        'right-hip': 9,
        'right-knee': 10,
        'right-shoulder': 11,
        'right-wrist': 12
    }
    bones: list[tuple[str, str]] = [
        ('head', 'left-shoulder'),
        ('head', 'right-shoulder'),
        ('left-shoulder', 'left-elbow'),
        ('left-shoulder', 'left-hip'),
        ('left-elbow', 'left-wrist'),
        ('right-shoulder', 'right-elbow'),
        ('right-shoulder', 'right-hip'),
        ('right-elbow', 'right-wrist'),
        ('left-hip', 'right-hip'),
        ('left-hip', 'left-knee'),
        ('left-knee', 'left-foot'),
        ('right-hip', 'right-knee'),
        ('right-knee', 'right-foot')
    ]

    # Draw nodes
    output = frame.copy()
    for fencer in keypoints:
        for node_name, node_idx in nodes.items():
            node_pos = fencer[node_idx]
            node_pos_ints = (int(node_pos[0]), int(node_pos[1]))
            node_color: tuple[int, int, int]
            if node_name.startswith('left-'):
                node_color = (255, 0, 255)
            elif node_name.startswith('right-'):
                node_color = (0, 0, 255)
            else:
                node_color = (0, 255, 0)
            output = cv2.circle(output, node_pos_ints, keypoint_radius, node_color, -1)
    
    # Draw bones
    for fencer in keypoints:
        for bone in bones:
            start_idx = nodes[bone[0]]
            start_pos = fencer[start_idx]
            start_pos_ints = (int(start_pos[0]), int(start_pos[1]))

            end_idx = nodes[bone[1]]
            end_pos = fencer[end_idx]
            end_pos_ints = (int(end_pos[0]), int(end_pos[1]))

            output = cv2.line(output, start_pos_ints, end_pos_ints, (0, 0, 127), bone_thickness)
    
    return output


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
    # cap: cv2.VideoCapture = get_stream(args)

    # while cap.isOpened():
    #     # Get the next frame
    #     ret, frame = cap.read()
    #     if not ret:
    #         print('Stream ended; closing...')
    #         break

    #     # Process frame with scorebox detection
    #     scorebox_classification, scorebox_boxes = scorebox_detector.detect_and_classify(frame, debug=False)
    #     print("Scorebox Boxes:", scorebox_boxes)

    #     # Process frame with fencer pose classification
    #     labeled_frame_fencers, fencer_boxes, fencer_keypoints = fencer_pose_classifier.evaluate_on_input(frame)  # Process the same frame
    #     print("Fencer Boxes:", fencer_boxes)
    #     print("Fencer keypoints:", fencer_keypoints)

    #     # Result of the scorebox classification (left, right, both or none)
    #     print(f"Scorebox Classification: {scorebox_classification}")

    #     # Draw bounding boxes on the original frame for the scorebox
    #     annotated_frame = draw_bounding_boxes(frame, scorebox_boxes)
    #     annotated_frame = draw_keypoints(annotated_frame, fencer_keypoints)
    #     # Resize the frame to reduce width and height by half
    #     frame_resized = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    #     # Display the frame and wait for 1/30th of a second
    #     cv2.imshow('stream', frame_resized)
    #     # cv2.waitKey(int(1000 / 30))
    #     if scorebox_classification != cv2_common.NO_SIDE:
    #         cv2.waitKey(0) # Pause when a valid classification is found
    #     else:
    #         cv2.waitKey(0) # Continue running

    # cap.release()
    # cv2.destroyAllWindows()

    img = cv2.imread("datasets/fencers/valid/images/0_0_jpg.rf.be88a72c7ba6419c6e1d3e95fbed280e.jpg")
    height, width = img.shape[:2]

    keypoints_raw = [
        [(0.3425408854166666, 0.40098203703703705), (0.3277820833333333, 0.4162481481481481), (0.3387653645833333, 0.4272397222222222), (0.32400661458333335, 0.430292962962963), (0.32558291666666667, 0.4777829629629629), (0.35112156250000004, 0.4534975), (0.37034229166666666, 0.4480016666666666), (0.32469307291666666, 0.4644941666666667), (0.3387653645833333, 0.4718219444444444), (0.32160401041666664, 0.49991157407407405), (0.31817171874999994, 0.5408247222222221), (0.35729963541666665, 0.4919731481481481), (0.35489708333333325, 0.5334970370370369)],
        [(0.5992749479166666, 0.4003426851851851), (0.6130040104166665, 0.4192726851851852), (0.5944697395833334, 0.4180513888888888), (0.6184956770833333, 0.4382026851851851), (0.6092285416666667, 0.4583539814814814), (0.59069421875, 0.4369813888888888), (0.57696515625, 0.4382026851851851), (0.6109446354166665, 0.4589646296296296), (0.5975588020833332, 0.4546900925925925), (0.61472015625, 0.4919394444444444), (0.6178092187499999, 0.5175864814814813), (0.5865755208333333, 0.4931606481481482), (0.5858890625, 0.526135462962963)]
    ]
    keypoints: list[list[tuple]] = []
    for fencer in keypoints_raw:
        fencer_points = []
        for point in fencer:
            fencer_points.append((point[0] * width, point[1] * height))
        keypoints.append(fencer_points)
    
    drawn_img = draw_keypoints(img, keypoints)
    cv2.imshow('img', drawn_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
