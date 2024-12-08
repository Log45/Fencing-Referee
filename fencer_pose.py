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

def train_new_model():
    """
    Train a new YOLO model.
    """
    # Load the base model (could be any of the versions: yolov8n.pt, yolov8s.pt, etc.)
    model = YOLO("yolov8n.pt")  # (Medium detect model) 
    # model = YOLO("yolov8n-pose.pt")  # (Medium pose model)
    
    trained_model = model.train(
        data="./datasets/data.yaml",        # Path to your data.yaml file
        epochs=100,                         # Number of epochs to train
        batch=-1,                           # Batch size
        imgsz=640,                          # Image size (default is 640x640)
        name="fencer_pose_yolov8_model",    # Experiment name
        save=True                           # Save the best and last models
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
        train_new_model()  # Train a new model
    else:
        print("Invalid choice. Please select either 1, 2, or 3.")

if __name__ == "__main__":
    main()

'''
#import the inference-sdk
from inference_sdk import InferenceHTTPClient

from inference import get_model
import supervision as sv
import cv2

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="yov2XDGTg2LCmTG6v8ey"
)

# infer on a local image
image_file = ".\\point_79-20241026T222021Z-001\\point_79\\frame360.jpg"
image = cv2.imread(image_file)
result = CLIENT.infer(image, model_id="fencer-pose/3")
'''


'''
# define the image url to use for inference
image_file = ".\\point_79-20241026T222021Z-001\\point_79\\frame360.jpg"
image = cv2.imread(image_file)

# Show image
image = cv2.resize(image, (960, 540))     # Resize image to fit on screen
cv2.imshow("raw_img", image)
cv2.waitKey(0)

# load a pre-trained yolov11 model
# Important: You must set up the api key in the environment first
model = get_model(model_id="fencer-pose/3")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0] #Can set up confidence threshold stuff here

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
'''