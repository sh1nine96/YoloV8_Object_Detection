# draw the bounding box on the image
#Draw labels
# Save the image

import cv2

def draw_predictions(image, result, model):
    """
    Draw bounding boxes and labels for the detected objects in the image.
    Args:
        image (numpy.ndarray): Input image on which to draw predictions.
        result: Predictions made by the YOLO model.
        model (YOLO): Loaded YOLO model.
    Returns:
        numpy.ndarray: Image with drawn bounding boxes and labels.
    """
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get the bounding box coordinates and convert them to integers
        class_id = int(box.cls[0])  # class ID of the detected object
        confidence = float(box.conf[0])  # confidence score of the detected object
        class_name = model.names[class_id]  # get the class name of the detected object
        label = f"{class_name} {confidence:.2f}"  # create a label for the detected object
        
        # draw bounding box
        cv2.rectangle(
            image,  # here we will draw the bounding box on the image
            (x1, y1),  # top-left corner of the bounding box
            (x2, y2),  # bottom-right corner of the bounding box
            (0, 255, 0),  # color of the bounding box (green)
            2
        )
        
        # draw label
        cv2.putText(
            image,  # here we will write the text on the image
            label,  # the text to be written
            (x1, y1 - 10),  # position of the text (write slightly above the bounding box)
            cv2.FONT_HERSHEY_SIMPLEX,  # font type
            0.6,  # font size
            (0, 255, 0),  # color of the text (green)
            2  # thickness of the text
        )
    
    return image