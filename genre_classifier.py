import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


class HybridGenreClassifier(nn.Module):
    """
    CNN-LSTM Hybrid Model for Audio Genre Classification
    """
    def __init__(self, num_classes, n_features, fixed_length, lstm_hidden_size=128, dropout=0.5):
        super(HybridGenreClassifier, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d((2, 2))
        self.lstm = nn.LSTM(input_size=fixed_length // 2, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), x.size(1), -1)  # Prepare for LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_and_evaluate_model(train_loader, test_loader, num_classes, n_features, fixed_length, epochs=20, learning_rate=0.001):
    """
    Train and evaluate the CNN-LSTM hybrid genre classifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridGenreClassifier(num_classes=num_classes, n_features=n_features, fixed_length=fixed_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.2f}")

    torch.save(model.state_dict(), "hybrid_genre_classifier.pth")
    print("Trained model saved to 'hybrid_genre_classifier.pth'.")
    return model


def probabilistic_genre_classification(model, inputs, label_encoder):
    """
    Perform probabilistic genre classification.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        genre_probabilities = {label: prob for label, prob in zip(label_encoder.classes_, probabilities[0])}
        return genre_probabilities
