import logging
import cv2

logger = logging.getLogger(__name__)


class FeatureProcessor(object):
    def __init__(self, local_descriptor='SIFT', dictionary_size=5):
        if local_descriptor == 'SIFT':
            self._feature_detector = cv2.xfeatures2d.SIFT_create()
            logger.info("Initialized SIFT detector")
        elif local_descriptor == 'ORB':
            self._feature_detector = cv2.ORB_create()
            logger.info("Initialized ORB detector")
        else:
            logger.error("Invalid local descriptor {}, defaulting to SIFT".format(local_descriptor))
            self._feature_detector = cv2.xfeatures2d.SIFT_create()

        # global image descriptor
        self._dictionary_size = dictionary_size
        self._bow = cv2.BOWKMeansTrainer(self._dictionary_size)

        # feature matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        self._feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        logger.info("Initialized FeatureProcessor")

    def get_local_descriptors(self, image):
        keypoints, descriptors = self._feature_detector.detectAndCompute(image, None)
        logger.info("Extracted local descriptor")
        return [keypoints, descriptors]

    def get_global_descriptor(self, local_descriptors):
        for descriptor in local_descriptors:
            self._bow.add(descriptor)
        return self._bow.cluster()

    def match_features(self, descriptors1, descriptors2):
        if descriptors1 is None or descriptors2 is None:
            return [], [], 0.
        matches = self._feature_matcher.knnMatch(descriptors1, descriptors2, k=2)
        positive_matches = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                positive_matches.append(i)
        num_positive_matches = len(positive_matches)
        num_matches = len(matches)
        similarity_score = float(num_positive_matches) / float(num_matches)
        logger.info("Matched features, got {}/{} positive matches. Similarity score = {}".format(num_positive_matches,
                                                                                                 num_matches,
                                                                                                 similarity_score))
        return matches, positive_matches,  similarity_score
