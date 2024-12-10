from ultralytics import YOLO
import cv2
from cv2.typing import MatLike
import numpy as np

class FencerPoseClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the classifier by loading a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model. If None, the model must be trained before use.
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

        # Initialize variables to store previous positions of the fencers
        self.previous_left_x = None
        self.previous_right_x = None

    def load_model(self, model_path):
        """
        Load a pre-trained YOLO model.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = YOLO(model_path)
        print(f"Model loaded from {model_path}")

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
    
    def fit_xy_to_original_size(self, keypoints: list[float], original_size: tuple[int, int]) -> list[int]:
        # Extract bounding box coordinates
        result = []
        for keypoint in keypoints:
            x, y= map(int, keypoint[:2])
            
            # Adjust bounding box to the original image size
            height, width = original_size
            x = int(x * width / 640)
            y = int(y * height / 640)
            
            result.append([x, y])
            
        return result

    def evaluate_on_input(self, input_path, save_output=False):
        """
        Evaluate the model on a single image or a video.

        Args:
            input_path (str or np.ndarray): Path to the image, video, or numpy array for evaluation.
            save_output (bool): Whether to save the output image or video with annotations.

        Returns:
            MatLike: The labeled frame(s) or None if processing a video.
        """
        if not self.model:
            raise ValueError("No model loaded. Please load a model first.")

        labeled_frame = None  # Initialize variable for labeled frame(s)
        left_boxes = [] # To store left fencer bounding box information
        left_keypoints = [] # To store left fencer keypoint information
        right_boxes = [] # To store right fencer bounding box information
        right_keypoints = [] # To store right fencer keypoint information
        # Movement calculation
        left_movement = 0
        right_movement = 0
        
        # Check if the input is a numpy array
        if isinstance(input_path, np.ndarray):
            img = input_path
            if img is None or img.ndim not in [2, 3]:
                raise ValueError("Invalid numpy array. Expected a valid image array.")

            # Get original size (height, width)
            original_size = img.shape[:2]
            
            # Resize the image to 640x640 (YOLOv8 input size)
            img_resized = cv2.resize(img, (640, 640))

            # Perform inference
            results = self.model(img_resized)
            for result in results:
                #print(result.boxes)  # Print detection boxes

                # Check which fencer is on the left
                if result.boxes.xyxy.tolist()[0][0] < result.boxes.xyxy.tolist()[1][0]:
                    left = 0
                    right = 1
                else:
                    left = 1
                    right = 0

                # Left fencer
                # Stretch bounding box to match the size of the original image
                left_bbox_xyxy_rescaled = self.fit_xyxy_to_original_size(result.boxes.xyxy.tolist()[left], original_size)
                # Save detection box info
                left_current_box = [left_bbox_xyxy_rescaled, result.boxes.conf.tolist()[left]]
                left_boxes.append(left_current_box)
                # Save keypoints info
                left_keypoints_rescaled = self.fit_xy_to_original_size(result.keypoints.xy.tolist()[left], original_size)
                left_keypoints.append(left_keypoints_rescaled)

                # Right fencer
                # Stretch bounding box to match the size of the original image
                right_bbox_xyxy_rescaled = self.fit_xyxy_to_original_size(result.boxes.xyxy.tolist()[right], original_size)
                # Save detection box info
                right_current_box = [right_bbox_xyxy_rescaled, result.boxes.conf.tolist()[right]]
                right_boxes.append(right_current_box)
                # Save keypoints info
                right_keypoints_rescaled = self.fit_xy_to_original_size(result.keypoints.xy.tolist()[right], original_size)
                right_keypoints.append(right_keypoints_rescaled)

                # ---------- Calculate Movement ------------
                # Calculate movement (delta_x) for both fencers
                left_center_x = (left_bbox_xyxy_rescaled[0] + left_bbox_xyxy_rescaled[2]) / 2
                right_center_x = -1 * (right_bbox_xyxy_rescaled[0] + right_bbox_xyxy_rescaled[2]) / 2

                # Calculate movement for left fencer
                if self.previous_left_x is not None:
                    left_movement = left_center_x - self.previous_left_x

                # Calculate movement for right fencer
                if self.previous_right_x is not None:
                    right_movement = right_center_x - self.previous_right_x

                # Update previous positions for the next frame
                self.previous_left_x = left_center_x
                self.previous_right_x = right_center_x

                #labeled_frame = result.plot()  # Annotated image
                if save_output:
                    result.save(filename="result.jpg")  # Save the annotated image
        else:
            raise ValueError("Unsupported file type")

        return left_boxes, left_keypoints, right_boxes, right_keypoints, left_movement, right_movement


    def train_model(self, dataset_yaml, epochs=100, batch_size=-1, model_name="fencer_pose_model", img_size=640):
        """
        Train a new YOLO model.

        Args:
            dataset_yaml (str): Path to the dataset YAML file.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training. -1 uses default.
            model_name (str): Name for the trained model.
            img_size (int): Image size for training (default is 640x640).
        """
        # Load the base model
        self.model = YOLO("yolo11n-pose.pt")

        # Train the model
        self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            name=model_name,
            save=True
        )

        print(f"Training completed. Model saved as '{model_name}'")

    def export_model(self, format="onnx"):
        """
        Export the model to a different format.

        Args:
            format (str): Format to export the model to (e.g., 'onnx', 'torchscript').
        """
        if not self.model:
            raise ValueError("No model loaded. Please load a model first.")

        self.model.export(format=format)
        print(f"Model exported in {format} format")

def main():
    fencer_classifier = FencerPoseClassifier()

    print("Choose an option:")
    print("1. Load a pre-trained model and evaluate it on an image.")
    print("2. Train a new model.")
    
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        model_path = input("Enter the path to the pre-trained model (e.g., 'best.pt'): ")
        fencer_classifier.load_model(model_path)
        image_path = input("Enter the path to the image for evaluation (e.g., 'image.jpg'): ")
        fencer_classifier.evaluate_on_input(image_path, save_output=True)
    elif choice == "2":
        dataset_yaml = input("Enter the path to the dataset YAML file: ")
        fencer_classifier.train_model(dataset_yaml)
    else:
        print("Invalid choice. Please select either 1 or 2.")

if __name__ == "__main__":
    main()