from ultralytics import YOLO
from ultralytics.engine.results import Results
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

    def load_model(self, model_path):
        """
        Load a pre-trained YOLO model.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = YOLO(model_path)
        print(f"Model loaded from {model_path}")

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
        boxes: list[tuple[list[float], float, float]] = [] # To store fencer bounding box information
        keypoints: list[list[tuple[float, float]]] = [] # To store fencer keypoint information
        
        # Check if the input is a numpy array
        if isinstance(input_path, np.ndarray):
            img = input_path
            if img is None or img.ndim not in [2, 3]:
                raise ValueError("Invalid numpy array. Expected a valid image array.")

            # Resize the image to 640x640 (YOLOv8 input size)
            img_resized = cv2.resize(img, (640, 640))

            # Perform inference
            results: Results = self.model(img_resized)
            for result in results:
                #print(result.boxes)  # Print detection boxes
                # Save detection box info
                this_box = [result.boxes.xyxy.tolist()[0], result.boxes.conf.tolist()[0], result.boxes.cls.tolist()[0]]
                boxes.append(this_box)
                # Save keypoints info
                keypoints.append(result.keypoints.xy.tolist()[0])

                labeled_frame = result.plot()  # Annotated image
                if save_output:
                    result.save(filename="result.jpg")  # Save the annotated image

        # Check if the input is an image or a video
        elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process as an image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {input_path}")

            # Resize the image to 640x640 (YOLOv8 input size)
            img_resized = cv2.resize(img, (640, 640))

            # Perform inference
            results = self.model(img_resized)
            for result in results:
                print(result.boxes)  # Print detection boxes
                labeled_frame = result.plot()  # Annotated image
                if save_output:
                    result.save(filename="result.jpg")  # Save the annotated image

        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process as a video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video from path: {input_path}")

            # Prepare video writer if saving output
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video
                out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 640))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached or error reading frame.")
                    break

                # Resize the frame to 640x640
                frame_resized = cv2.resize(frame, (640, 640))

                # Perform inference
                results = self.model(frame_resized)
                for result in results:
                    print(result.boxes)  # Print detection boxes
                    labeled_frame = result.plot()  # Annotated frame

                    # Display the frame
                    cv2.imshow("Video", labeled_frame)
                    if save_output:
                        out.write(labeled_frame)  # Write frame to output video

                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            if save_output:
                out.release()
            cv2.destroyAllWindows()
            labeled_frame = None  # Labeled frames for video are displayed, not returned
        else:
            raise ValueError("Unsupported file type. Provide an image (.jpg, .png) or video (.mp4, .avi).")

        # Perform final transformations back to source size
        img_size = img.shape[:2]
        boxes = list(map(lambda box_data: (self.fit_xyxy_to_original_size(box_data[0], img_size), box_data[1], box_data[2]), boxes))
        keypoints = list(map(lambda points: self.fit_points_to_original_size(points, img_size), keypoints))

        return labeled_frame, boxes, keypoints


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
    
    def fit_points_to_original_size(self, points: list[tuple[float, float]], original_size: tuple[int, int]) -> list[int]:
        # Extract points
        points_ints = map(lambda tup: (int(tup[0]), int(tup[1])), points)

        # Scale points
        height, width = original_size
        return list(map(lambda tup: (int(tup[0] * width / 640), int(tup[1] * height / 640)), points_ints))

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