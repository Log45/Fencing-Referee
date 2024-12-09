# conda env config vars set ROBOFLOW_API_KEY=yov2XDGTg2LCmTG6v8ey
# conda activate c:\Users\alynf\OneDrive\Documents\Fencing-Referee\.conda

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def load_and_evaluate_model(model_path, image_path):
    """
    Load a pre-trained model, evaluate it on an image, and display the results.

    Args:
        model_path (str): Path to the pre-trained model.
        image_path (str): Path to the image to be evaluated.
    """
    # Load the pre-trained model
    model = YOLO(model_path)

    # Load the image
    img = cv2.imread(image_path)
    
    # Resize the image to 640x640 (YOLOv8 input size)
    img_resized = cv2.resize(img, (640, 640))  # Resize image to 640x640
    
    # Perform inference (predict) on the image
    results = model(img_resized)
    for result in results:
        print(result.boxes)  # Print detection boxes
        result.show()  # Display the annotated image
        result.save(filename="result.jpg")  # Save annotated image

def train_new_model(mode):
    """
    Train a new YOLO model.
    mode (str): Mode for processing ('fencers' or 'scoreboxes').
    """
    # Load the base model (could be any of the versions: yolov8n.pt, yolov8s.pt, etc.)
    if mode == 'fencers':
        model = YOLO("yolo11n-pose.pt")  # (Medium detect model) 

        trained_model = model.train(
            data="./datasets/fencers_data.yaml",        # Path to your data.yaml file
            epochs=10,                                  # Number of epochs to train
            batch=-1,                                   # Batch size
            imgsz=640,                                  # Image size (default is 640x640)
            name="fencer_pose_yolov11_model",           # Experiment name
            save=True                                   # Save the best and last models
        )
        
    elif mode == 'scoreboxes':
        model = YOLO("yolo11n.pt")

        trained_model = model.train(
            data="./datasets/scoreboxes_data.yaml",        # Path to your data.yaml file
            epochs=10,                                     # Number of epochs to train
            batch=-1,                                      # Batch size
            imgsz=640,                                     # Image size (default is 640x640)
            name="scorebox_detection_yolov11_model",       # Experiment name
            save=True                                      # Save the best and last models
        )

    # Export the model (optional, to use in different formats)
    model.export(format="onnx")  # Available formats: "onnx", "torchscript", "coreml", etc.

def main():
    """
    Main function to allow the user to choose between loading a pre-trained model and evaluating it on an image.
    """
    print("Choose an option:")
    print("1. Load a pre-trained model and show its metrics.")
    print("2. Evaluate a pre-trained model on an image.")
    print("3. Train a new model.")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == "1":
        model_path = input("Enter the path to the pre-trained model (e.g., 'best.pt'): ")
        load_and_evaluate_model(model_path, image_path=None)  # Only evaluate model metrics, no image
    elif choice == "2":
        model_path = input("Enter the path to the pre-trained model (e.g., 'best.pt'): ")
        image_path = input("Enter the path to the image for evaluation (e.g., 'image.jpg'): ")
        load_and_evaluate_model(model_path, image_path)  # Evaluate the model on the specified image
    elif choice == "3":
        print("Are you trainining a model for:")
        print("1. Scorebox detection.")
        print("2. Fencer and saber detection and keypoints.")
        choice = input("Enter your choice (1 or 2): ")
        # Train a new model
        if choice == "1":
            train_new_model("scoreboxes")
        elif choice == "2":
            train_new_model("fencers")
        else:
            print("Invalid choice. Please select either 1 or 2.")
    else:
        print("Invalid choice. Please select either 1, 2, or 3.")

if __name__ == "__main__":
    main()
