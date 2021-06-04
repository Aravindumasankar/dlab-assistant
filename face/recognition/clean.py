import os
from face.face_recognition_cli import image_files_in_folder

for class_dir in os.listdir('train'):
    print(class_dir)
    count = 0
    for img_path in image_files_in_folder(os.path.join('train', class_dir)):
        if os.path.getsize(img_path) < 20 * 1024:
            print(img_path)
            count += 1
            os.remove(img_path)
print(count)