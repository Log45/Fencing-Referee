# Saber Fencing Hit Allocation with YOLO Pose

## Getting Started

To run this pipeline, follow these steps:

```python
pip install -r requirements.txt
python main.py # run the pipeline
```

## Repo Overview

Here is an overview of the important parts of the project:

- `main.py`: Runs the mainloop of the fencing scoring pipeline
    - `-m`, `--mode`: Inference mode [optional]; either "webcam" or "file"
    - `-f`, `--file`: Path to the file you want to run inference on [optional]
    - `--headless`: Use if you are running in a container or with opencv-python-headless
- `fencer_pose.py`: Holds the Class for our Fencing Pose Estimator
- `scorebox_classifier.py`: Holds the Class for detecting points based on the fencing scorebox
- `yolo_scorebox_classifier.py`: Holds the Class for both detecting scoreboxes as well as detecting points based on the detected scorebox
- `nn_pose_classifier.py`: Holds the Class for reading pose estimations and classifying which action they correspond to
- `classifier_data.py`: Holds functions to take pose classification labels and convert them to a readable format
