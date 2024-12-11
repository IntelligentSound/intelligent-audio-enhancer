import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedGenreClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Model for Audio Genre Classification.
    """
    def __init__(self, num_classes, n_features, fixed_length, lstm_hidden_size=128, dropout=0.5):
        super(EnhancedGenreClassifier, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2, 2))
        self.lstm = nn.LSTM(input_size=fixed_length // 2, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convolutional layer
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = x.view(x.size(0), x.size(1), -1)  # Flatten for LSTM
        x, _ = self.lstm(x)  # Pass through LSTM
        x = x[:, -1, :]  # Take the last output of LSTM
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # Output layer
        return x


def load_model(model_path, num_classes, n_features, fixed_length, device="cpu"):
    """
    Load a pre-trained EnhancedGenreClassifier model.

    Parameters:
        model_path (str): Path to the saved model.
        num_classes (int): Number of output classes.
        n_features (int): Number of input features.
        fixed_length (int): Fixed length of input features.
        device (str): Device to load the model on.

    Returns:
        model: Loaded model.
    """
    model = EnhancedGenreClassifier(num_classes, n_features, fixed_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def predict_probabilities(model, inputs):
    """
    Predict genre probabilities for given inputs.

    Parameters:
        model (EnhancedGenreClassifier): Trained model.
        inputs (torch.Tensor): Input data.

    Returns:
        np.ndarray: Probabilities for each genre.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()
    return probabilities
