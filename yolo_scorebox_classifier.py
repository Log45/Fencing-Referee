import cv2
import cv2_common
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from scorebox_classifier import ScoreboxThresholdClassifier


class ScoreboxDetectorClassifier:
    _detector: YOLO

    def __init__(self, model_path: str):
        self._detector = YOLO(model_path)

    def crop_image_with_bbox(self, image, bbox, original_size):
        """
        Crop an image using bounding box coordinates, adjusted to the original image size.
        
        Args:
            image (numpy.ndarray): The input image.
            bbox (torch.Tensor): Bounding box in xyxy format (x_min, y_min, x_max, y_max).
            original_size (tuple): The original size of the image (height, width).
        
        Returns:
            numpy.ndarray: Cropped image.
        """        
        x_min, y_min, x_max, y_max = self.fit_xyxy_to_original_size(bbox, original_size)
        return image[y_min:y_max, x_min:x_max]
    
    def fit_xyxy_to_original_size(self, bbox: list[float], original_size: tuple[int, int]) -> list[int]:
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        
        # Adjust bounding box to the original image size
        height, width = original_size
        x_min = int(x_min * width / 640)
        y_min = int(y_min * height / 640)
        x_max = int(x_max * width / 640)
        y_max = int(y_max * height / 640)

        return [x_min, y_min, x_max, y_max]

    def detect_and_classify(self, img: MatLike, debug: bool = False):
        """
        Detect scoreboxes in an image using YOLO and classify using thresholding.
        
        Args:
            img (MatLike): The input image.
        
        Output: A tuple of the detection result and the labeled image.
        """

        # Get original size (height, width)
        original_size = img.shape[:2]
        
        # Resize for YOLO input
        img_resized = cv2.resize(img, (640, 640))  # Resize to 640x640 for YOLO input
    
        # Prepare a copy of the input image for labeling
        #img_labeled = img.copy()

        # Perform detection
        results = self._detector(img_resized, show=debug)

        # Initialize the classifier
        classifier = ScoreboxThresholdClassifier()

        # To store the bounding boxes
        boxes = []

        # Process each detected box
        classification: str = cv2_common.BOTH_SIDES
        for result in results:
            box: Boxes = result.boxes  # Boxes object for bounding box outputs
            # print(box.xyxy)

            # Skip boxes results that didn't detect anything
            if box.xyxy.shape[0] == 0:
                continue

            # Get general results
            bbox_xyxy = box.xyxy.tolist()[0]
            confidence = box.conf.tolist()[0]
            # print("XYXY:", bbox_xyxy)
            # print("Confidence:", confidence)

            # Stretch bounding box to match the size of the original image
            bbox_xyxy_rescaled = self.fit_xyxy_to_original_size(bbox_xyxy, original_size)

            # If confidence is above a threshold, proceed
            if confidence > 0.5:
                print(f"Detected box with confidence {confidence:.2f}")

                # Save the bounding box info
                this_box = [bbox_xyxy_rescaled, confidence]
                boxes.append(this_box)

                # Draw bounding box
                #img_labeled = cv2.rectangle(img_labeled, bbox_xyxy_rescaled[0:2], bbox_xyxy_rescaled[2:4], (0, 255, 0), 3)

                # Crop the detected bounding box region adjusted to the original image size
                cropped_img = self.crop_image_with_bbox(img, bbox_xyxy, original_size)

                # Display the cropped image
                if debug:
                    cv2.imshow("Cropped Image", cropped_img)

                # Classify the cropped region
                classification = classifier.classify(cropped_img, show_images=debug)
                print(f"Classification result: {classification}")
        
        return classification, boxes
    
    def train_new_model():
        """
        Train a new YOLO model.
        """
        # Load the base model (could be any of the versions: yolov8n.pt, yolov8s.pt, etc.)
        model = YOLO("yolo11n.pt")

        trained_model = model.train(
            data="./datasets/scoreboxes_data.yaml",        # Path to your data.yaml file
            epochs=100,                                    # Number of epochs to train
            batch=-1,                                      # Batch size
            imgsz=640,                                     # Image size (default is 640x640)
            name="scorebox_detection_yolov11_model",       # Experiment name
            save=True                                      # Save the best and last models
        )

        # Export the model (optional, to use in different formats)
        model.export(format="onnx")  # Available formats: "onnx", "torchscript", "coreml", etc.

if __name__ == "__main__":
    # User inputs
    print("Performing scorebox identification with YOLO and color threshold classification")
    model_path = input("Enter the path to the YOLO model (e.g., 'best.pt'): ")
    image_path = input("Enter the path to the image for evaluation (e.g., 'image.jpg'): ")
    
    # Perform detection and classification
    img = cv2.imread(image_path)
    detector_classifier = ScoreboxDetectorClassifier(model_path)
    detector_classifier.detect_and_classify(img, debug=True)
