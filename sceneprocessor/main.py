from . import utils, describer
import argparse
import logging
import os

import cv2

logging.basicConfig()
logger = logging.getLogger()


def main(args):
    image_describer = describer.Describer('SIFT')
    input_dir = os.path.realpath(args.input_dir)
    for filename in utils.load_sequence(input_dir):
        image = utils.load_image(os.path.join(input_dir, filename))
        keypoints = image_describer.extract_local_descriptors(image)
        print(keypoints)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Display informational output')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input sequence directory')
    parser.add_argument('--output-file', type=str, help="Path to output text file", default='output.txt')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    main(args)
