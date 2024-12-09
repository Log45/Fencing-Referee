'''
To be run on the imported roboflow data:
- Converts bounding boxes from (label, x center, y center, width, height) to (label, x min, y min, x max, y max)
- Removes the "occluded" (visibility) data from the keypoints
'''
import os
import glob
import shutil


def convert_xywh_to_xyxy(data):
    """
    Convert bounding box format from x, y, width, height to x_min, y_min, x_max, y_max.

    Args:
        data (list): List of label data including class_id, bounding box, and keypoints.

    Returns:
        tuple: Converted class_id, x_min, y_min, x_max, y_max, followed by the remaining keypoints.
    """
    # Extract the first 5 values: class_id, x_center, y_center, width, height
    class_id, x_center, y_center, width, height = data[:5]
    x_center, y_center, width, height = map(float, [x_center, y_center, width, height])

    # Convert to x_min, y_min, x_max, y_max
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)

    return class_id, x_min, y_min, x_max, y_max


def preprocess_data(input_file, output_file):
    """
    Preprocess the dataset by removing the extra data points (visibility flags or confidence scores)
    and converting bounding box format to x_min, y_min, x_max, y_max.

    Args:
        input_file (str): Path to the input dataset file.
        output_file (str): Path where the preprocessed dataset will be saved.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()  # Read all lines to process
        for i, line in enumerate(lines):
            # Split the line into individual values
            data = line.strip().split()

            '''
            # Skip lines with label 1
            if data[0] == "1":
                continue

            # Rename lines with label 2 to 1
            if data[0] == "2":
                data[0] = "1"
            '''

            # Skip scoreboxes
            if data[0] == "2":
                continue

            # Convert bounding box format
            class_id, x_min, y_min, x_max, y_max = data[:5]  # convert_xywh_to_xyxy(data)

            # Process the keypoints, removing the extra values (e.g., visibility/confidence)
            keypoints = []
            for i in range(5, len(data), 3):  # Each keypoint has 3 values (x, y, extra)
                keypoint_x = data[i]
                keypoint_y = data[i + 1]
                # We skip the extra value (e.g., visibility flag or confidence score
                keypoints.append(f"{keypoint_x} {keypoint_y}")

            # Reconstruct the line without the extra keypoint data
            processed_line = f"{class_id} {x_min} {y_min} {x_max} {y_max} " + " ".join(keypoints)

            outfile.write(processed_line + "\n")
            '''
            # Write the processed line to the file without a trailing newline on the last line
            if i < len(lines) - 1:
                outfile.write(processed_line + "\n")
            else:
                outfile.write(processed_line)
            '''
    print(f"Data preprocessing complete. Processed file saved as '{output_file}'.")


def process_and_manage_labels(base_dir):
    """
    Handle the directories for label processing:
    - If 'old_labels' exists, process the files inside and store the output in 'labels'.
    - If 'old_labels' does not exist, process the files in 'labels', move them to 'old_labels', and save the output in 'labels'.

    Args:
        base_dir (str): Path to the base directory containing the 'labels' folder.
    """
    labels_dir = os.path.join(base_dir, 'labels')
    old_labels_dir = os.path.join(base_dir, 'old_labels')

    # Determine input and output directories
    if os.path.exists(old_labels_dir):
        print(f"'old_labels' directory exists. Processing files from '{old_labels_dir}'.")
        input_dir = old_labels_dir
    else:
        print(f"'old_labels' directory does not exist. Processing files from '{labels_dir}' and creating 'old_labels'.")
        os.makedirs(old_labels_dir, exist_ok=True)
        for file in glob.glob(os.path.join(labels_dir, "*.txt")):
            shutil.move(file, old_labels_dir)  # Move existing labels to 'old_labels'
        input_dir = old_labels_dir

    # Process files and save output in 'labels'
    os.makedirs(labels_dir, exist_ok=True)  # Ensure 'labels' directory exists
    for input_file in glob.glob(os.path.join(input_dir, "*.txt")):
        output_file = os.path.join(labels_dir, os.path.basename(input_file))
        preprocess_data(input_file, output_file)

    print(f"Processed labels saved in '{labels_dir}'.")


# Directories to process
directories = [
    './datasets/test',
    './datasets/train',
    './datasets/valid'
]

# Process all directories
for directory in directories:
    process_and_manage_labels(directory)