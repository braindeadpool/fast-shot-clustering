import logging
import natsort
import os

import cv2

logging.basicConfig()
logger = logging.getLogger(__name__)


def load_sequence(sequence_dir):
    file_sequence = []
    if os.path.isdir(sequence_dir):
        for entry in os.scandir(sequence_dir):
            if entry.is_file() and entry.name.endswith('.jpg'):
                file_sequence.append(entry.name)
        file_sequence = natsort.natsorted(file_sequence)
    else:
        logger.error("Invalid sequence directory")
    return file_sequence


def load_image(image_path, grayscale=False):
    image = None
    if os.path.isfile(image_path):
        try:
            image = cv2.imread(image_path)
            if grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            logger.error("Could not load image {}".format(image_path))
    else:
        logger.error("Invalid image path")
    return image
