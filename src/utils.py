#Paths
# Folders
# common helper functions

from pathlib import Path
import logging
import os

def get_project_paths():
    project_dir = Path(__file__).resolve().parent.parent

    paths = {
        "project": project_dir,
        "images": project_dir / "images",
        "outputs": project_dir / "outputs",
        "models": project_dir / "models"
    }

    paths["outputs"].mkdir(exist_ok=True)

    return paths


# here we will do the logging

def setup_logger():
    """ 
    Configure and return a logger

    Why?
    Logging is preferred over print()
    bcz it provides timestamp, log levels and can save messages to file

    Returns
    logging.logger
        configured logger object
    """

    logging.basicConfig(
        level=logging.INFO,  # INFO, warning, error
        format="%(asctime)s | %(level_name)s | %(message)s"  #21-07-2026 14:58:30, message --> Model loaded succcesfully
    )

    return logging.getLogger(__name__)


def validate_image(image_path):
    """
    it will check if image exists,
    Parameter --> image_path: Path

    Raise: FileNotFoundError if the image does not exists

    Instead of OpenCV fail later, we check the file before processing
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f'Image not found: {image_path}'
        )


def get_all_images(image_folder):
    """
    Return all supported images
    Raise FileNotFoundError if the image is not found
    """

    image_extensions = ["*.jpg", "*jpeg", "*.png"]
    image_paths = []
    for extenion in image_extensions:
        image_paths.extend(
            image_folder.glob(extenion)
        )


    if len(image_paths) == 0:
        raise FileNotFoundError(
            "No images found inside image folder"
        )
    return image_paths