import cv2
import os
from matplotlib import pyplot as plt
import albumentations as A


def resize(image, bboxes, classID):
    transform = A.Compose(
        [A.LongestMaxSize(p=1, max_size=1000, always_apply=True)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['classID'], min_visibility=0.15),
    )
    transformed = transform(image=image, bboxes=bboxes, classID=classID)

    return transformed


def save_label(classID, bboxes, file_path):
    line = []
    for id, box in zip(classID, bboxes):
        line.append(str(id) + ' ' + ' '.join([str(x) for x in box]) + '\n')

    file = open(file_path, 'w', encoding="utf-8")
    file.writelines(line)


def load_class(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    return [x[:-1] for x in lines]


def load_label(file_path):
    file = open(file_path, 'r', encoding="utf-8")
    lines = file.readlines()

    classID = []
    bboxes = []
    for line in lines:
        line = line[:-1].split(" ")
        classID.append(int(line[0]))
        bboxes.append([float(x) for x in line[1:]])

    return classID, bboxes


imgFolder = "./images"  # Image folder
labelFolder = "./labels"  # Label folder
classFile = "./classes.txt"  # Class file

classNames = load_class(classFile)  # load class name

for root, dirs, files in os.walk(imgFolder):
    for file in files:
        if file.endswith(".jpg") | file.endswith(".png") | file.endswith(".JPG"):
            imgPath = imgFolder + '/' + file
            txtPath = labelFolder + '/' + file[:-4] + ".txt"
            print('Read: ', imgPath, txtPath)

            classID, bboxes = load_label(txtPath)

            # load img
            image = cv2.imread(imgPath)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # show before
            # visualize(image, bboxes, classID, classNames)

            transformed = resize(image, bboxes, classID)
            os.remove(imgPath)
            os.remove(txtPath)
            cv2.imwrite(imgPath, transformed['image'])
            save_label(transformed['classID'], transformed['bboxes'], txtPath)
            print('ReSize Save: ', imgPath, txtPath)
            print()
