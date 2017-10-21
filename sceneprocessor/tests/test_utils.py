import logging
import os
import unittest

from ..utils import load_sequence, load_image, logger

import numpy as np


class TestUtils(unittest.TestCase):
    def setUp(self):
        self._this_dir = os.path.dirname(os.path.realpath(__file__))
        self._sequence_dir = os.path.join(self._this_dir, 'data')

    def test_load_sequence(self):
        true_sequence = ['000001.jpg', '000003.jpg', '000005.jpg', '000007.jpg', '000008.jpg', '000009.jpg',
                         '000010.jpg', '000098.jpg', '000099.jpg', '000100.jpg']
        loaded_sequence = load_sequence(self._sequence_dir)
        self.assertEqual(len(loaded_sequence), 10)
        for i, filename in enumerate(loaded_sequence):
            self.assertEqual(true_sequence[i], filename)

    def test_load_image(self):
        # Test if it checks for invalid path
        image = load_image(self._sequence_dir)
        self.assertIsNone(image)
        self.assertLogs(logger, logging.ERROR)
        # Test if it checks for invalid file
        image = load_image(os.path.join(self._sequence_dir, 'not_an_image.abcd'))
        self.assertIsNone(image)
        self.assertLogs(logger, logging.ERROR)
        # Test if an actual image loads
        image = load_image(os.path.join(self._sequence_dir, '000001.jpg'))
        self.assertIsInstance(image, np.ndarray)
        np.testing.assert_array_equal(image.shape, (360, 640, 3))


if __name__ == '__main__':
    unittest.main()
