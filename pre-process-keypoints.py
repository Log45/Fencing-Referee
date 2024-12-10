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
    class_id, x_center, y_center, width, height = data[:5]
    x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    return class_id, x_min, y_min, x_max, y_max


def preprocess_data(input_file, output_file, mode):
    """
    Preprocess the dataset by removing extra data and converting bounding box format.

    Args:
        input_file (str): Path to the input dataset file.
        output_file (str): Path where the preprocessed dataset will be saved.
        mode (str): Mode for processing ('fencers' or 'scoreboxes').
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        for line in lines:
            data = line.strip().split()
            if mode == 'fencers':
                if data[0] == "1":  # Skip sabers
                    continue
                if data[0] == "2":  # Skip scoreboxes
                    continue
                class_id, x_min, y_min, x_max, y_max = data[:5]  # convert_xywh_to_xyxy(data)
                keypoints = []
                for i in range(5, len(data), 3):
                    keypoint_x = data[i]
                    keypoint_y = data[i + 1]
                    keypoints.append(f"{keypoint_x} {keypoint_y}")
                processed_line = f"{class_id} {x_min} {y_min} {x_max} {y_max} " + " ".join(keypoints)
            elif mode == 'scoreboxes':
                if data[0] == "0":  # Skip fencers
                    continue
                if data[0] == "1":  # Skip sabers
                    continue
                if data[0] == "2":
                    data[0] = "0"  # Rename scoreboxes
                processed_line = f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} "
            else:
                continue
            outfile.write(processed_line + "\n")
    print(f"Data preprocessing complete. Processed file saved as '{output_file}'.")


def copy_images(input_dir, output_dir):
    """
    Copy all image files from input_dir to output_dir.

    Args:
        input_dir (str): Source directory for images.
        output_dir (str): Destination directory for images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_file in glob.glob(os.path.join(input_dir, "*.*")):
        shutil.copy2(img_file, os.path.join(output_dir, os.path.basename(img_file)))


def delete_directory(directory):
    """
    Delete a directory if it exists.

    Args:
        directory (str): Path to the directory to delete.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Deleted directory: {directory}")


def process_and_manage_labels_and_images(mode, base_dir):
    """
    Handle the directories for label processing, image moving, and cleanup.

    Args:
        mode (str): Mode for processing ('fencers' or 'scoreboxes').
        base_dir (str): Base directory ('train', 'valid', 'test').
    """
    labels_dir = os.path.join('./datasets', mode, base_dir, 'labels')
    images_dir = os.path.join('./datasets', mode, base_dir, 'images')
    unprocessed_labels_dir = os.path.join('./datasets/unprocessed', base_dir, 'labels')
    unprocessed_images_dir = os.path.join('./datasets/unprocessed', base_dir, 'images')
    base_labels_dir = os.path.join('./datasets', base_dir, 'labels')
    base_images_dir = os.path.join('./datasets', base_dir, 'images')

    # Process labels
    if not os.path.exists(unprocessed_labels_dir):
        os.makedirs(unprocessed_labels_dir, exist_ok=True)
        for file in glob.glob(os.path.join(base_labels_dir, "*.txt")):
            shutil.move(file, unprocessed_labels_dir)
    os.makedirs(labels_dir, exist_ok=True)
    for input_file in glob.glob(os.path.join(unprocessed_labels_dir, "*.txt")):
        output_file = os.path.join(labels_dir, os.path.basename(input_file))
        preprocess_data(input_file, output_file, mode)

    # Process images based on mode
    if not os.path.exists(unprocessed_images_dir):
        os.makedirs(unprocessed_images_dir, exist_ok=True)
        copy_images(base_images_dir, unprocessed_images_dir)
    if mode == "fencers":
        copy_images(unprocessed_images_dir, images_dir)
    elif mode == "scoreboxes":
        copy_images(unprocessed_images_dir, os.path.join('./datasets/scoreboxes', base_dir, 'images'))
    else:  # unprocessed
        copy_images(unprocessed_images_dir, os.path.join('./datasets/unprocessed', base_dir, 'images'))

    # Delete original base_dir
    base_dir_path = os.path.join('./datasets', base_dir)
    delete_directory(base_dir_path)

    print(f"Processed labels and images saved in '{os.path.join('./datasets', mode, base_dir)}'.")


# Directories to process
directories = ['train', 'valid', 'test']

# Process all directories for both modes
for mode in ["fencers", "scoreboxes"]:
    for directory in directories:
        process_and_manage_labels_and_images(mode, directory)