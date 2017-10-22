from . import utils, featureprocessor
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger()


def main(args):
    feature_processor = featureprocessor.FeatureProcessor('ORB')
    input_dir = os.path.realpath(args.input_dir)
    file_sequence = utils.load_sequence(input_dir)[:args.frame_limit]

    similarity_scores = []
    prev_descriptors = None

    outlier_threshold = 0.4

    for i, filename in tqdm(enumerate(file_sequence)):
        image = utils.load_image(os.path.join(input_dir, filename))
        keypoints, descriptors = feature_processor.get_local_descriptors(image)
        matches, positive_matches, similarity_score = feature_processor.match_features(descriptors,
                                                                                       prev_descriptors)
        similarity_scores.append(similarity_score)

        if similarity_score < outlier_threshold:
            print("Scene change detected at frame {}".format(filename))

        prev_descriptors = descriptors

    if args.plot:
        xlabels = np.arange(1, len(file_sequence) + 1, 10)
        plt.plot(similarity_scores, color='g', marker='+', linestyle='None')
        plt.xticks(xlabels)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Display informational output')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input sequence directory')
    parser.add_argument('--output-file', type=str, help="Path to output text file", default='output.txt')
    parser.add_argument('--plot', action='store_true', help='Plot metrics')
    parser.add_argument('--frame-limit', type=int, default=None, help='Number of frames to limit the sequence to')
    parser.add_argument('--window-size', type=int, default=40,
                        help='Maximum of past frames to use for computing rolling stats')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    main(args)
