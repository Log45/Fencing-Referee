"""
Script to convert anylabeling labels to YOLO format.

Taken from https://github.com/ThijsCol/Anylabeling-LabelMe-json-to-yolo-txt?tab=readme-ov-file under MIT License.
"""
from jsontoyolo import json_to_yolo
import os
import cv2

def segregate_labels(converted_dir: str, output_dir: str) -> None:
    """
    Segregate the images from the txt labels.

    Args:
    converted_dir (str): The directory containing the converted labels (txt+image).
    output_dir (str): The directory to hold the fully converted dataset.
        It is assumed that converted_dir contains a 'train' and 'validate' directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0
    for d in ['train', 'validate']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        os.makedirs(os.path.join(output_dir, d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, d, 'labels'), exist_ok=True)
        
        for f in os.listdir(os.path.join(converted_dir, d)):
            if f.endswith('.txt'):
                os.rename(os.path.join(converted_dir, d, f), os.path.join(output_dir, d, 'labels', f))
            else:
                os.rename(os.path.join(converted_dir, d, f), os.path.join(output_dir, d, 'images', f))
                file_count += 1
            
    
    print("Label Segregation Complete.")
    print(f"Processed {file_count} images.")

def validate_labels(dataset_dir: str) -> None:
    """Remove any images that do not have corresponding labels and vice versa"""
    for d in ['train', 'validate']:
        for f in os.listdir(os.path.join(dataset_dir, d, 'images')):
            if not os.path.exists(os.path.join(dataset_dir, d, 'labels', f.replace('.jpg', '.txt'))):
                os.remove(os.path.join(dataset_dir, d, 'images', f))
                
        for f in os.listdir(os.path.join(dataset_dir, d, 'labels')):
            if not os.path.exists(os.path.join(dataset_dir, d, 'images', f.replace('.txt', '.jpg'))):
                os.remove(os.path.join(dataset_dir, d, 'labels', f))
        
    print("Validation Complete." if len(os.listdir(os.path.join(dataset_dir, 'train', 'images'))) == len(os.listdir(os.path.join(dataset_dir, 'train', 'labels'))) and len(os.listdir(os.path.join(dataset_dir, 'validate', 'images'))) == len(os.listdir(os.path.join(dataset_dir, 'validate', 'labels'))) else "Validation Failed.")

def generate_data_list(dataset_dir: str) -> None:
    """Generate a list of images for training and validation"""
    for d in ['train', 'validate']:
        with open(os.path.join(dataset_dir, d + '.txt'), 'w') as f:
            for img in os.listdir(os.path.join(dataset_dir, d, 'images')):
                f.write(os.path.join("./", d, 'images', img).replace('\\', '/') + '\n')
    
    print("Data List Generated.")

def resize_images(dataset_dir: str, size: tuple) -> None:
    """Resize all images in the dataset"""
    for d in ['train', 'validate']:
        for img in os.listdir(os.path.join(dataset_dir, d, 'images')):
            image = cv2.imread(os.path.join(dataset_dir, d, 'images', img))
            image = cv2.resize(image, size)
            cv2.imwrite(os.path.join(dataset_dir, d, 'images', img), image)
    
    print(f"Images Resized to {size[0]}x{size[1]}.")

def main():
    for d in os.listdir('./labels'):
        if os.path.isdir('./labels/' + d):
            json_to_yolo('./labels/' + d, './yolo/', {"scorebox": 0}, 0.2)
    
    segregate_labels('./yolo', './dataset')
    validate_labels('./dataset')
    generate_data_list('./dataset')
            
if __name__ == '__main__':
    main()