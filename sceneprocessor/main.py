from . import utils, featureprocessor
import argparse
import logging
import os

import matplotlib.pyplot as plt
import natsort
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
    clustering_threshold = 0.2

    # Dictionary storing frame of scene change and corresponding descriptors  for labeling.
    last_label = 'A'
    scene_change_to_descriptors = {}
    scene_change_to_label = {file_sequence[0]: last_label}

    for i, filename in tqdm(enumerate(file_sequence)):
        image = utils.load_image(os.path.join(input_dir, filename))
        _, descriptors = feature_processor.get_local_descriptors(image)
        matches, positive_matches, similarity_score = feature_processor.match_features(descriptors,
                                                                                       prev_descriptors)
        similarity_scores.append(similarity_score)

        if len(matches) > 0 and similarity_score < outlier_threshold:
            print("Scene change detected at frame {}".format(filename))
            label_assigned = False
            if filename in scene_change_to_label:
                continue
            max_score = 0.
            for prev_filename in scene_change_to_descriptors:
                matches, _, score = feature_processor.match_features(descriptors,
                                                                     scene_change_to_descriptors[prev_filename])
                if score > clustering_threshold and score > max_score:
                    scene_change_to_label[filename] = scene_change_to_label[prev_filename]
                    max_score = score
                    label_assigned = True

            scene_change_to_descriptors[filename] = descriptors
            if not label_assigned:
                last_label = chr(ord(last_label) + 1)
                scene_change_to_label[filename] = last_label

        prev_descriptors = descriptors

    print("keyframe, scene-id")
    for filename in natsort.natsorted(scene_change_to_label.keys()):
        print("{}, {}".format(filename, scene_change_to_label[filename]))

    if args.plot:
        x_labels = np.arange(1, len(file_sequence) + 1, 10)
        plt.plot(similarity_scores, color='g', marker='+', linestyle='None')
        plt.xticks(x_labels)
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
