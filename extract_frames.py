import cv2
import os
from pathlib import Path

def extract_frames(file_path: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_stem = Path(file_path).stem
    vidcap = cv2.VideoCapture(file_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print(f'FPS: {fps}')
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        if not (count % fps) or (count % fps == fps // 5) or (count % fps == fps // 4) or (count % fps == fps // 3) or (count % fps == fps // 2) or (count % fps == fps // 1.5) or (count % fps == fps // 1.1):
            print(f'Reading frame {count} {success}')
            cv2.imwrite(f"{save_dir}/{file_stem}_{count}.jpg", image)     # save frame as JPEG file      
        count += 1

dir = Path("./Fencing-Dataset/Edited_Clips")
save_dir = Path("./Fencing-Dataset/extracted_frames")
for d in os.listdir(dir):
    if os.path.isdir(dir / d):
        for f in os.listdir(dir / d):
            if os.path.isfile(dir / d / f):
                extract_frames(str(dir / d / f), str(save_dir / d / f))
            elif os.path.isdir(dir / d / f):
                for ff in os.listdir(dir / d / f):
                    if os.path.isfile(dir / d / f / ff):
                        extract_frames(str(dir / d / f / ff), str(save_dir / d / f / ff))
            

