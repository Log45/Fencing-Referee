import cv2
import numpy as np
from ultralytics import YOLO
from scorebox_classifier import ScoreboxThresholdClassifier

def crop_image_with_bbox(image, bbox, original_size):
    """
    Crop an image using bounding box coordinates, adjusted to the original image size.
    
    Args:
        image (numpy.ndarray): The input image.
        bbox (torch.Tensor): Bounding box in xyxy format (x_min, y_min, x_max, y_max).
        original_size (tuple): The original size of the image (height, width).
    
    Returns:
        numpy.ndarray: Cropped image.
    """
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    
    # Adjust bounding box to the original image size
    height, width = original_size
    x_min = int(x_min * width / 640)
    y_min = int(y_min * height / 640)
    x_max = int(x_max * width / 640)
    y_max = int(y_max * height / 640)
    
    return image[y_min:y_max, x_min:x_max]

def detect_and_classify(model_path, image_path):
    """
    Detect scoreboxes in an image using YOLO and classify using thresholding.
    
    Args:
        model_path (str): Path to the YOLO model.
        image_path (str): Path to the input image.
    """
    # Initialize YOLO model
    yolo_model = YOLO(model_path)

    # Load the original image
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Get original size (height, width)
    
    # Resize for YOLO input
    img_resized = cv2.resize(img, (640, 640))  # Resize to 640x640 for YOLO input

    # Perform detection
    results = yolo_model(img_resized, show=True)

    # Initialize the classifier
    classifier = ScoreboxThresholdClassifier()

    # Process each detected box
    for result in results:
        box = result.boxes  # Boxes object for bounding box outputs
        print(box)

        bbox_xyxy = box.xyxy.tolist()[0]
        confidence = box.conf.tolist()[0]
        print("XYXY:", bbox_xyxy)
        print("Confidence:", confidence)

        # If confidence is above a threshold, proceed
        if confidence > 0.5:
            print(f"Detected box with confidence {confidence:.2f}")

            # Crop the detected bounding box region adjusted to the original image size
            cropped_img = crop_image_with_bbox(img, bbox_xyxy, original_size)

            # Display the cropped image
            cv2.imshow("Cropped Image", cropped_img)

            # Classify the cropped region
            classification = classifier.classify(cropped_img, show_images=True)
            print(f"Classification result: {classification}")

if __name__ == "__main__":
    # User inputs
    print("Performing scorebox identification with YOLO and color threshold classification")
    model_path = input("Enter the path to the YOLO model (e.g., 'best.pt'): ")
    image_path = input("Enter the path to the image for evaluation (e.g., 'image.jpg'): ")
    
    # Perform detection and classification
    detect_and_classify(model_path, image_path)
