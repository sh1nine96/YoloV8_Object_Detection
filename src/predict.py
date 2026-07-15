# load the model
# predictions of the objects in the image

from ultralytics import YOLO

def load_model(model_name = "yolov8n.pt"):
    """
    Load the YOLO model.
    Args:
        model_name (str): Name of the YOLO model to load. Default is 'yolov8n.pt'.
    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_name)
    return model


def predict_image(model, image_path, conf=0.5):
    """
    Predict objects in the image using the YOLO model.
    Args:
        model (YOLO): Loaded YOLO model.
        image_path (str): Path to the input image.
        conf (float): Confidence threshold for predictions. Default is 0.5.
        save (bool): Whether to save the output image with bounding boxes. Default is True.
    Returns:
        results: Predictions made by the model.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False, #this will suppress the verbose output of the model
    )
    return results