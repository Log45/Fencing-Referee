'''
Main program for processing videos and video streams
'''

import cv2
import cv2_common
from argparse import ArgumentParser, Namespace
from yolo_scorebox_classifier import ScoreboxDetectorClassifier


## Constants

SCOREBOX_MODEL_PATH = 'trained_models/scorebox_detect/scorebox_detect.pt'


## Code

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


def main():
    # Parse command-line args
    parser = ArgumentParser(prog="Fencing-Referee", description="Automatically scores fencing bouts")
    parser.add_argument('-m', '--mode', choices=['webcam', 'file'], required=False)
    parser.add_argument('-f', '--filename', required=False)
    args = parser.parse_args()

    # Load model
    scorebox_detector = ScoreboxDetectorClassifier(SCOREBOX_MODEL_PATH)

    # Open video stream
    cap: cv2.VideoCapture = get_stream(args)

    while cap.isOpened():
        # Get the next frame
        ret, frame = cap.read()
        if not ret:
            print('Stream ended; closing...')
            break

        # Process
        scorebox_classification, labeled_frame = scorebox_detector.detect_and_classify(frame, debug=True)
        print(scorebox_classification)

        # Display the frame and wait for 1/30th of a second
        cv2.imshow('stream', labeled_frame)
        # cv2.waitKey(int(1000 / 30))
        if scorebox_classification != cv2_common.NO_SIDE:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
