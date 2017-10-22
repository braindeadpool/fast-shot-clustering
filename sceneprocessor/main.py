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

    # statistics for outlier detection in similarity scores
    prev_mean = 0.
    prev_std = 0.
    for i, filename in tqdm(enumerate(file_sequence)):
        image = utils.load_image(os.path.join(input_dir, filename))
        keypoints, descriptors = feature_processor.get_local_descriptors(image)
        matches, positive_matches, similarity_score = feature_processor.match_features(descriptors,
                                                                                       prev_descriptors)
        similarity_scores.append(similarity_score)

        if i == 0:
            current_mean = similarity_score
            current_std = 0.
        else:
            current_mean = prev_mean + (similarity_score - prev_mean) / (i + 1)
            current_std = np.sqrt((prev_std + (similarity_score - prev_mean) * (similarity_score - current_mean)) / i)

        if similarity_score != 0 and similarity_score < current_mean - 6 * current_std:
            print("Scene change detected at frame {}".format(filename))

        prev_descriptors = descriptors
        prev_mean = current_mean
        prev_std = current_std

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
