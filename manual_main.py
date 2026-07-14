import cv2
# for reading, writing and manipulating images. Drawing bounding boxes, etc.
from ultralytics import YOLO
# pretrained YOLO model for object detection, prediction, training, validation, etc.
import matplotlib.pyplot as plt
# it will be used to display the image with bounding boxes
from pathlib import Path
# "F:\Personal_Projects\360digiTMG\YoloV8_Object_Detection\images\street.jpg"
# Path("images")/street.jpg


PROJECT_DIR = Path(__file__).parent
IMAGES_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Loading pretrained YOLO nano model
model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n model

# print("Model loaded successfully!")
# print("Model summary:")
# model.info()  # print model summary
# print(model)


# Read the input image
image_path = IMAGES_DIR / "street.jpg"

image = cv2.imread(str(image_path))


# loop through every detection in the results

