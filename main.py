import cv2
import matplotlib.pyplot as plt
from src.utils import get_project_paths
from src.predict import load_model, predict_image
from src.visualize import draw_predictions

#get project paths
paths = get_project_paths()

# load model
model = load_model()  # load a pretrained YOLOv8n model

#input image path
image_path = paths["images"] / "street.jpg"

# read image
image = cv2.imread(str(image_path))


# perform detection
results = predict_image(model, image_path, conf=0.5)

# draw detections on the image

annotated_image = draw_predictions(image, results[0], model)


# Convert BGR to RGB for displaying with matplotlib
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


# display the annotated image
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image_rgb)
plt.title("Detected Objects")
plt.axis('off')
plt.show()
