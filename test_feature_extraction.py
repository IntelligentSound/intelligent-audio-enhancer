# test_feature_extraction.py

import unittest
from feature_extraction import extract_features
import numpy as np

class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features_valid_file(self):
        # Provide a valid audio file path
        file_path = 'path_to_valid_audio_file.au'
        features = extract_features(file_path)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (40, 100))

    def test_extract_features_invalid_file(self):
        # Provide an invalid audio file path
        file_path = 'invalid_file_path.au'
        features = extract_features(file_path)
        self.assertIsNone(features)

if __name__ == '__main__':
    unittest.main()
