import os
from pathlib import Path
import yaml
import numpy as np

def generate_training_data():
    """
    Read the pose classification dataset and return the data and classes.
    
    Reads the pose classification dataset from the datasets/pose_classification/labels.txt, which includes image paths and labels, and then
    read the ground truth pose labels from the datasets/fencers/train/labels, datasets/fencers/test/labels, and datasets/fencers/valid/labels directories to 
    generate the training data.
    
    Returns:
    data: np.array, shape (n_samples, n_features+1)
        The labeled data (X and y).
    classes: list
    
    """
    labels_path = Path("datasets/pose_classification/labels.txt")
    data_path = Path("datasets/pose_classification/data.yaml")
    
    test_path = Path("datasets/fencers/test/labels")
    train_path = Path("datasets/fencers/train/labels")
    val_path = Path("datasets/fencers/valid/labels")
    
    data_yaml = yaml.load(data_path.read_text(), Loader=yaml.FullLoader)
    classes = data_yaml["classes"]
    im_paths = []
    labels = []
    
    with open(labels_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            im_path, label_l, label_r = line.strip().split()
            im_paths.append(im_path)
            labels.append((int(label_l), int(label_r)))
    
    pose_label_paths = []
    pose_label_list = []
    
    test_labels = os.listdir(test_path)
    train_labels = os.listdir(train_path)
    val_labels = os.listdir(val_path)
    
    for label in test_labels:
        #print(label, label.partition("_jpg")[0])
        if label.partition("_jpg")[0] in im_paths:
            pose_label_paths.append(label)
            with open(test_path / label, "r") as f:
                sub = [label.partition("_jpg")[0]]
                for line in f:
                    splt = line.strip().split()
                    if splt[0] == "0":
                        sub.append((splt[1], splt[5:]))
                pose_label_list.append(sub)
                    
    
    for label in train_labels:
        if label.partition("_jpg")[0] in im_paths:
            pose_label_paths.append(label)
            with open(train_path / label, "r") as f:
                sub = [label.partition("_jpg")[0]]
                for line in f:
                    splt = line.strip().split()
                    if splt[0] == "0":
                        sub.append((splt[1], splt[5:]))
                pose_label_list.append(sub)
    
    for label in val_labels:
        if label.partition("_jpg")[0] in im_paths:
            pose_label_paths.append(label)
            with open(val_path / label, "r") as f:
                sub = [label.partition("_jpg")[0]]
                for line in f:
                    splt = line.strip().split()
                    if splt[0] == "0":
                        sub.append((splt[1], splt[5:]))
                pose_label_list.append(sub)
    
    file_to_label = {}
    for i, label in enumerate(pose_label_list):
        if len(label) == 3:
            d = {}
            x1 = label[1][0]
            x2 = label[2][0]
            if float(x1) > float(x2):
                d["left"] = [float(x) for x in label[1][1]]
                d["right"] = [float(x) for x in label[2][1]]
            else:
                d["left"] = [float(x) for x in label[2][1]]
                d["right"] = [float(x) for x in label[1][1]]
            file_to_label[label[0]] = d
        else:
            print("Invalid label: ", label)
            
    # print(file_to_label, len(file_to_label))
    
    data = []
    for key in file_to_label:
        left, right = labels[im_paths.index(key)]
        l_data = file_to_label[key]["left"]
        l_data.append(float(left))
        r_data = file_to_label[key]["right"]
        r_data.append(float(right))
        data.append(l_data)
        data.append(r_data)
    
    data = np.array(data)

    return data, classes
    