import random

import cv2
import os
from matplotlib import pyplot as plt
import albumentations as A

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


# img size
def cv_size(img):
    return tuple(img.shape[1::-1])


# yolo to pixel x,y
def yolobbox2bbox(x, y, w, h, img_w, img_h):
    x1, y1 = x * img_w - w * img_w / 2, y * img_h - h * img_h / 2
    x2, y2 = x * img_w + w * img_w / 2, y * img_h + h * img_h / 2
    return x1, y1, x2, y2


# draw box
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    img_w, img_h = cv_size(img)
    x, y, w, h = bbox
    x1, y1, x2, y2 = yolobbox2bbox(x, y, w, h, img_w, img_h)
    x_min, x_max, y_min, y_max = int(x1), int(x2), int(y1), int(y2)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


# show img
def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


# load label
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


# load class name
def load_class(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    return [x[:-1] for x in lines]


# transform Img
def transform(image, bboxes, classID):
    w, h = cv_size(image)

    transform = A.Compose(
        [A.HorizontalFlip(p=0.4),
         A.RGBShift(p=0.3),
         A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.3, 0.3)),
         A.RandomGamma(p=0.5, gamma_limit=(60, 140)),
         A.ShiftScaleRotate(p=0.5),
         A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.4),
         A.Blur(p=0.5, blur_limit=3),
         A.CenterCrop(p=0.1, width=int(w / 2), height=int(h / 2)),
         A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
         A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.2, src_radius=300),
         ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['classID'], min_visibility=0.15),
    )
    transformed = transform(image=image, bboxes=bboxes, classID=classID)

    return transformed


# Resize Img
def resize(image, bboxes, classID):

    transform = A.Compose(
        [A.LongestMaxSize(p=1, max_size=640, always_apply=True)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['classID'], min_visibility=0.15),
    )
    transformed = transform(image=image, bboxes=bboxes, classID=classID)

    return transformed


def save_label(classID, bboxes, file_path):
    line = []
    for id, box in zip(classID, bboxes):
        line.append(str(id) + ' ' + ' '.join([str(x) for x in box])+'\n')

    file = open(file_path, 'w', encoding="utf-8")
    file.writelines(line)


imgFolder = "./images"  # Image folder
labelFolder = "./labels"  # Label folder
classFile = "./classes.txt"  # Class file

classNames = load_class(classFile)  # load class name

# loop img
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

            # transform
            for i in range(4):
                transformed = transform(image, bboxes, classID)

                # show after
                # visualize(transformed['image'], transformed['bboxes'], transformed['classID'], classNames)

                SaveImgPath = f"{imgFolder}/{file[:-4]}_{i}{file[-4:]}"
                SaveLabelPath = f"{labelFolder}/{file[:-4]}_{i}.txt"
                print('Save: ', SaveImgPath, SaveLabelPath)

                cv2.imwrite(SaveImgPath, transformed['image'])
                save_label(transformed['classID'], transformed['bboxes'], SaveLabelPath)

