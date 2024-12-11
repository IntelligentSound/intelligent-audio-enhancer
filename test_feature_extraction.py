import unittest
import os
from feature_extraction import extract_features_with_openl3, separate_audio

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.test_audio = "test_audio.wav"  # Replace with a valid path to a test audio file
        self.invalid_audio = "invalid_audio.wav"

    def test_extract_features_valid_audio(self):
        features = extract_features_with_openl3(self.test_audio)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 512)  # Check OpenL3 feature dimensions

    def test_extract_features_invalid_audio(self):
        with self.assertLogs(level='ERROR'):
            features = extract_features_with_openl3(self.invalid_audio)
            self.assertIsNone(features)

    def test_separate_audio_valid(self):
        separated_files = separate_audio(self.test_audio)
        self.assertIsNotNone(separated_files)
        self.assertIn("vocals", separated_files)
        self.assertTrue(os.path.exists(separated_files["vocals"]))

    def test_separate_audio_invalid(self):
        separated_files = separate_audio(self.invalid_audio)
        self.assertIsNone(separated_files)

if __name__ == '__main__':
    unittest.main()
