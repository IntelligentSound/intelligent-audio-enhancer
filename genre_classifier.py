# genre_classifier.py
# genre_classifier.py

# genre_classifier.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pydub import AudioSegment
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from feature_extraction import extract_features, extract_features_from_signal, add_noise, augment_time_stretch
import joblib
import logging
from torch.utils.tensorboard import SummaryWriter
import warnings
import sys
from model import EnhancedGenreClassifier  # Import from model.py
# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/genre_classification_experiment')

# Define the enhanced neural network model with batch normalization
class EnhancedGenreClassifier(nn.Module):
    def __init__(self, num_classes, n_features, fixed_length, dropout=0.5):
        super(EnhancedGenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        # Modify pooling to only pool the height dimension
        self.pool = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)

        # Calculate the size after convolutions and pooling
        self.conv_output_size = self._get_conv_output(n_features, fixed_length)
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output(self, n_features, fixed_length):
        """
        Computes the size of the output from the convolutional layers.
        """
        with torch.no_grad():
            # Adjust fixed_length to match the data (fixed_length=100)
            dummy_input = torch.zeros(1, 1, n_features, fixed_length)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return int(np.prod(x.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_audio_files(base_path):
    """
    Loads audio files and corresponding genres from the dataset directory.

    Parameters:
        base_path (str): Path to the base directory containing genre folders.

    Returns:
        audio_files (list): List of file paths to audio files.
        genres (list): List of genre labels corresponding to each audio file.
    """
    audio_files = []
    genres = []
    for genre in os.listdir(base_path):
        genre_path = os.path.join(base_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.lower().endswith(('.wav', '.mp3', '.au', '.flac')):
                    audio_files.append(os.path.join(genre_path, file))
                    genres.append(genre)
    return audio_files, genres

def load_data(audio_files, genres, num_classes, fixed_length=100):
    """
    Loads and preprocesses data, including feature extraction and data augmentation.

    Parameters:
        audio_files (list): List of file paths to audio files.
        genres (list): List of genre labels.
        num_classes (int): Number of unique genres.
        fixed_length (int): Number of time frames for feature extraction.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        label_encoder (LabelEncoder): Fitted label encoder.
    """
    X = []
    y = []
    augmented_X = []
    augmented_y = []

    label_encoder = LabelEncoder()

    # Encode labels
    encoded_labels = label_encoder.fit_transform(genres)

    for idx, (file_path, label) in enumerate(zip(audio_files, encoded_labels)):
        print(f"Processing file {idx + 1}/{len(audio_files)}")
        logging.info(f"Processing file {idx + 1}/{len(audio_files)}: {file_path}")
        # Extract features from the original file
        features = extract_features(file_path, fixed_length=fixed_length)
        if features is not None:
            X.append(features)
            y.append(label)

            # Data Augmentation: Add noise
            try:
                audio_segment = AudioSegment.from_file(file_path)
                y_audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                y_audio /= (np.max(np.abs(y_audio)) + 1e-9)
                y_noisy = add_noise(y_audio)
                features_noisy = extract_features_from_signal(y_noisy, fixed_length=fixed_length, sr=audio_segment.frame_rate)
                if features_noisy is not None:
                    augmented_X.append(features_noisy)
                    augmented_y.append(label)
                    
                    logging.info(f"Added noisy augmentation for {file_path}")
            except Exception as e:
                print(f"Error processing noise augmentation for {file_path}: {e}")
                logging.error(f"Error processing noise augmentation for {file_path}: {e}")
                continue

            # Data Augmentation: Time Stretching
            try:
                y_stretched = augment_time_stretch(y_audio, rate=1.05)  # Ensure rate is a keyword argument
                features_stretched = extract_features_from_signal(y_stretched, fixed_length=fixed_length, sr=audio_segment.frame_rate)
                if features_stretched is not None:
                    augmented_X.append(features_stretched)
                    augmented_y.append(label)
                    
                    logging.info(f"Added time-stretched augmentation for {file_path}")
            except Exception as e:
                print(f"Error processing time stretching for {file_path}: {e}")
                logging.error(f"Error processing time stretching for {file_path}: {e}")
                continue

    if not X:
        print("No features extracted. Please check your dataset and feature extraction process.")
        logging.error("No features extracted. Please check your dataset and feature extraction process.")
        sys.exit(1)

    # Combine original and augmented data
    X = np.array(X + augmented_X)
    y = np.array(y + augmented_y)

    print(f"Total samples after augmentation: {X.shape[0]}")
    logging.info(f"Total samples after augmentation: {X.shape[0]}")

    # Handle class imbalance using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(X.shape[0], -1), y)

    print(f"After oversampling: {X_resampled.shape[0]} samples")
    logging.info(f"After oversampling: {X_resampled.shape[0]} samples")

    # **Removed PCA**
    # # Apply PCA for dimensionality reduction
    # pca = PCA(n_components=50)
    # X_pca = pca.fit_transform(X_resampled)
    # joblib.dump(pca, 'pca.joblib')
    # logging.info("PCA fitted and saved to 'pca.joblib'.")

    # # Stratified train-test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    # )

    # # Use original features without PCA
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled.reshape(X_resampled.shape[0], 40, 100), y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    logging.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Convert to tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()

    # Reshape for CNN: [batch_size, channels, n_features, time_steps]
    X_train = X_train.unsqueeze(1)  # [batch_size, 1, 40, 100]
    X_test = X_test.unsqueeze(1)    # [batch_size, 1, 40, 100]

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader, label_encoder

def main():
    try:
        print("Starting main function.")
        logging.info("Starting main function.")

        # Define the base path to your dataset
        base_path = 'C:/Users/bomah/Documents/Training_datasets/genres'  # Replace with the actual path to your dataset
        print(f"Dataset path set to: {base_path}")
        logging.info(f"Dataset path set to: {base_path}")

        if not os.path.exists(base_path):
            print(f"Dataset path '{base_path}' does not exist. Please check the path.")
            logging.error(f"Dataset path '{base_path}' does not exist.")
            return

        # Load audio files and genres
        audio_files, genres = load_audio_files(base_path)
        print("Loaded audio files and genres.")
        logging.info("Loaded audio files and genres.")

        if len(audio_files) == 0:
            print("No audio files found. Please check your dataset directory.")
            logging.error("No audio files found.")
            return

        print(f"Found {len(audio_files)} audio files across {len(set(genres))} genres.")
        logging.info(f"Found {len(audio_files)} audio files across {len(set(genres))} genres.")

        # Determine the number of classes
        num_classes = len(set(genres))
        print(f"Number of classes: {num_classes}")
        logging.info(f"Number of classes: {num_classes}")

        # Define fixed_length (must match the one used in feature_extraction.py)
        fixed_length = 100
        print(f"Fixed length for feature extraction: {fixed_length}")
        logging.info(f"Fixed length for feature extraction: {fixed_length}")

        # Load data and get data loaders
        print("Loading data and preparing data loaders...")
        train_loader, test_loader, label_encoder = load_data(
            audio_files, genres, num_classes, fixed_length=fixed_length
        )
        print("Data loaded and data loaders prepared.")
        logging.info("Data loaded and data loaders prepared.")

        # Save the label encoder
        joblib.dump(label_encoder, 'label_encoder.joblib')
        logging.info("Label encoder saved to 'label_encoder.joblib'")
        print("Label encoder saved to 'label_encoder.joblib'")

        # Determine n_features from the data
        n_features = train_loader.dataset.tensors[0].shape[2]  # 40 MFCCs
        print(f"Number of features (MFCCs): {n_features}")
        logging.info(f"Number of features (MFCCs): {n_features}")

        # Initialize the model, criterion, optimizer, and scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        logging.info(f"Using device: {device}")

        model = EnhancedGenreClassifier(num_classes, n_features, fixed_length, dropout=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        print("Model, criterion, optimizer, and scheduler initialized.")
        logging.info("Model, criterion, optimizer, and scheduler initialized.")

        # Cross-Validation Setup
        print("Setting up cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = []
        y = []
        for features, label in train_loader.dataset:
            X.append(features.numpy().flatten())
            y.append(label.numpy())
        X = np.array(X)
        y = np.array(y)
        print("Cross-validation data prepared.")
        logging.info("Cross-validation data prepared.")

        fold_accuracies = []

        for fold, (train_ids, test_ids) in enumerate(skf.split(X, y)):
            print(f'FOLD {fold + 1}')
            logging.info(f'FOLD {fold + 1}')

            # Sample elements according to the current fold
            X_train, X_test = X[train_ids], X[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]
            print(f"Processing Fold {fold + 1}")
            logging.info(f"Processing Fold {fold + 1}: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

            # Convert to tensors
            X_train_tensor = torch.tensor(X_train).float()
            y_train_tensor = torch.tensor(y_train).long()
            X_test_tensor = torch.tensor(X_test).float()
            y_test_tensor = torch.tensor(y_test).long()

            # Reshape for CNN
            X_train_tensor = X_train_tensor.view(-1, 1, n_features, fixed_length)
            X_test_tensor = X_test_tensor.view(-1, 1, n_features, fixed_length)

            # Create DataLoaders for the current fold
            train_dataset_fold = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset_fold = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True, num_workers=4)
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False, num_workers=4)

            # Initialize a new model for each fold
            model_fold = EnhancedGenreClassifier(num_classes, n_features, fixed_length, dropout=0.5).to(device)
            optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler_fold = optim.lr_scheduler.StepLR(optimizer_fold, step_size=10, gamma=0.1)
            print(f"Initialized model for Fold {fold + 1}.")
            logging.info(f"Initialized model for Fold {fold + 1}.")

            # Train the model
            num_epochs = 20
            for epoch in range(num_epochs):
                model_fold.train()
                running_loss = 0.0
                for inputs, labels in train_loader_fold:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer_fold.zero_grad()
                    outputs = model_fold(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_fold.step()
                    running_loss += loss.item()
                scheduler_fold.step()
                avg_loss = running_loss / len(train_loader_fold)
                logging.info(f"Fold {fold +1}, Epoch {epoch +1}/{num_epochs}, Loss: {avg_loss:.4f}")
                writer.add_scalar(f'Fold{fold +1}/Training Loss', avg_loss, epoch +1)
                print(f"Fold {fold +1}, Epoch {epoch +1}/{num_epochs}, Loss: {avg_loss:.4f}")

                # Evaluate the model on the test set
                model_fold.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader_fold:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model_fold(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                logging.info(f"Fold {fold +1}, Epoch {epoch +1}/{num_epochs}, Accuracy: {accuracy:.2f}%")
                writer.add_scalar(f'Fold{fold +1}/Test Accuracy', accuracy, epoch +1)
                print(f"Fold {fold +1}, Epoch {epoch +1}/{num_epochs}, Accuracy: {accuracy:.2f}%")
                fold_accuracies.append(accuracy)

        # Average accuracy across all folds
        avg_accuracy = np.mean(fold_accuracies)
        print(f'Average Cross-Validation Accuracy: {avg_accuracy:.2f}%')
        logging.info(f'Average Cross-Validation Accuracy: {avg_accuracy:.2f}%')

        # Save the final model (trained on the last fold)
        torch.save(model_fold.state_dict(), 'enhanced_genre_classifier.pth')
        logging.info("Model trained and saved to 'enhanced_genre_classifier.pth'")
        print("Model trained and saved to 'enhanced_genre_classifier.pth'")

        writer.close()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
