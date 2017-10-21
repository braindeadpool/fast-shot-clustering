import logging
import cv2

logger = logging.getLogger(__name__)


class Describer(object):
    def __init__(self, local_descriptor='SIFT'):
        if local_descriptor == 'SIFT':
            self._feature_detector = cv2.xfeatures2d.SIFT_create()
            logger.info("Initialized SIFT detector")
        elif local_descriptor == 'ORB':
            self._feature_detector = cv2.ORB_create()
            logger.info("Initialized ORB detector")
        else:
            logger.error("Invalid local descriptor {}, defaulting to SIFT".format(local_descriptor))
            self._feature_detector = cv2.xfeatures2d.SIFT_create()

    def extract_local_descriptors(self, image):
        keypoints = self._feature_detector.detect(image, None)
        return keypoints
