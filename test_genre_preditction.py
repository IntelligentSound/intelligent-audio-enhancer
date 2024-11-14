# test_genre_prediction.py

import unittest
import torch
from model import EnhancedGenreClassifier
from ai_equalizer import predict_genre
from sklearn.preprocessing import LabelEncoder
import joblib

class TestGenrePrediction(unittest.TestCase):
    def setUp(self):
        # Setup a dummy model and label encoder
        self.num_classes = 10
        self.n_features = 40
        self.fixed_length = 100
        self.model = EnhancedGenreClassifier(self.num_classes, self.n_features, self.fixed_length)
        self.model.eval()
        self.device = torch.device("cpu")
        self.label_encoder = LabelEncoder()
        # Dummy label encoder
        self.label_encoder.classes_ = np.array(['Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5',
                                               'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10'])

    def test_predict_genre_valid_features(self):
        # Create dummy features
        features = torch.randn(40, 100)
        # Mock file path
        file_path = 'dummy_file.au'
        genre = predict_genre(self.model, self.label_encoder, file_path, self.device, fixed_length=100)
        # Since the model is untrained, the prediction is arbitrary
        self.assertIn(genre, self.label_encoder.classes_)

    def test_predict_genre_invalid_features(self):
        # Pass None as features
        genre = predict_genre(self.model, self.label_encoder, None, self.device, fixed_length=100)
        self.assertIsNone(genre)

if __name__ == '__main__':
    unittest.main()
