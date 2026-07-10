"""
Workflow

Image -> Yolo Model -> Detect objects -> Draw bounding boxes -> Save image
"""

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

# print(type(image))
# print("Image shape:", image.shape)

## Diaplay the input image
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# The Actual input is in RGB format, but OpenCV reads the image in BGR format. 
# and matplotlib expects the image in RGB format.
# So we need to convert it to RGB format before displaying it.
# plt.figure(figsize=(10, 9))
# plt.imshow(image_rgb)  
# plt.title("Input Image")
# plt.axis('off')
# plt.show()

results = model.predict(
        source= str(image_path),  # path to the input image
        conf=0.5,
        save = True
          )  # predict objects in the image with confidence threshold of 0.5

# print(type(results))
# print(results)

# it will save the output image with bounding boxes in the "runs/detect/predict" directory by default.


# Lets access the first prediction result from the results list

result = results[0]
# print(type(result))

# because we have only one image, so we can access the first result from the results list.

# Lets display the detected Objects'
# The boxes attribute store every detected object in the image. 
# It is a list of Box objects, where each Box object contains the coordinates of the bounding box, 
# the class label, and the confidence score.

# print("Detected Objects:")
# print(result.boxes)  # print the detected objects' bounding boxes

# lets count the number of detected objects in the image
num_objects = len(result.boxes)
print(f"Number of detected objects: {num_objects}")

# Lets inspect the first detected object in the image
# Each Detection contains the following attribute:
# Bounding box coordinates (x1, y1, x2, y2), class ID, confidence score, etc.

box = result.boxes[0]  # access the first detected object
print("First Detected Object:")
print(box)


# Lets extract the class and coinfidence score of the first detected object
class_id = int(box.cls[0])  # class ID of the first detected object
confidence = float(box.conf[0])  # confidence score of the first detected object
print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")



# now we will extract class names of the detected objects in the image.
# The model.names attribute contains a dictionary that maps class IDs to class names.

# print(model.names) # print the class names of the model

class_name = model.names[class_id]  # get the class name of the first detected object
print(f"Class Name: {class_name}")


# display every detected object along with its class name and confidence score

print("Detected Objects:")
print("*" * 30)
for box in result.boxes:
    class_id = int(box.cls[0])  # class ID of the detected object
    confidence = float(box.conf[0])  # confidence score of the detected object
    class_name = model.names[class_id]  # get the class name of the detected object
    print(f"Class Name: {class_name}, Confidence: {confidence:.2f}")

print("*" * 30)

