import cv2
import matplotlib.pyplot as plt
from src.predict import load_model, predict_image
from src.visualize import draw_predictions
from src.utils import (
    get_project_paths,
    setup_logger
)


#Wrap the entire programme in one try-except


try: 
    #get project paths
    paths = get_project_paths()

    # logger
    logger = setup_logger()
    logger.info("project started") # date time | INFO | project started

    # load model
    model = load_model()  # load a pretrained YOLOv8n model
    logger.info("YOLO model loaded successfully") # date time | INFO | YOLO model loaded successfully

    #input image path
    image_paths = list(paths["images"].glob("*.jpg"))  # get all jpg images in the images folder
    print(f'Found {len(image_paths)} images in the images folder.')
    # print(f'Using image: {image_paths[0]}')  # use the first image for testing
    # for i in image_paths:
    #     print(f'Using image: {i}')  # use the first image for testing

    for image_path in image_paths:
        print(f'Using image: {image_path.name}')  # use the first image for testing



    for image_path in image_paths:
        #read image
        image = cv2.imread(str(image_path))
        # sometimes image is there but corrupted or may contain invalid data
        # OpenCV returns None
        if image is None:
            logger.error(
                f'unable to read image: {image_path.name}'
            )
            continue
        logger.info(f'loaded image: {image_path.name}') # date time | INFO | loaded image: street.jpg

        #Detect objects in the image
        results = predict_image(
            model,
            image_path,
            conf=0.5
        )
        logger.info("Object Detection completed")

        # draw detections on the image
        annotated_image = draw_predictions(
            image, 
            results[0], 
            model
            )

        logger.info("Bounding boxes drawn")
        

        # Convert BGR to RGB for displaying with matplotlib
        annotated_image = cv2.cvtColor(
            annotated_image, 
            cv2.COLOR_BGR2RGB
        )

        #Display the annotated image
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.title(f"Detected Objects in {image_path.name}")
        plt.axis('off')
        plt.show()

        logger.info("programme finished successfully")

except Exception as error:
    logger.exception(error)


    # logger.info
    # HW --> logger.warning(), logger.error()

# # read image
# image = cv2.imread(str(image_path))


# # perform detection
# results = predict_image(model, image_path, conf=0.5)

# # draw detections on the image

# annotated_image = draw_predictions(image, results[0], model)


# # Convert BGR to RGB for displaying with matplotlib
# annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


# # display the annotated image
# plt.figure(figsize=(12, 8))
# plt.imshow(annotated_image_rgb)
# plt.title("Detected Objects")
# plt.axis('off')
# plt.show()

