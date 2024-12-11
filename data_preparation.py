from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data_with_separation(audio_files, genres, fixed_length=100, use_separation=True):
    """
    Load data, optionally separate audio into components, and extract features.

    Parameters:
        audio_files (list): List of audio file paths.
        genres (list): List of genres corresponding to the audio files.
        fixed_length (int): Number of time frames to standardize.
        use_separation (bool): Whether to separate audio ingredients using Spleeter.

    Returns:
        tuple: Features (X), labels (y), and label encoder.
    """
    X = []
    y = []

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(genres)

    for idx, (file_path, label) in enumerate(zip(audio_files, encoded_labels)):
        print(f"Processing file {idx + 1}/{len(audio_files)}: {file_path}")
        try:
            if use_separation:
                separated_files = separate_audio(file_path)
                for ingredient, ingredient_path in separated_files.items():
                    features = extract_features_with_openl3(ingredient_path, fixed_length=fixed_length)
                    if features is not None:
                        X.append(features)
                        y.append(label)
            else:
                features = extract_features_with_openl3(file_path, fixed_length=fixed_length)
                if features is not None:
                    X.append(features)
                    y.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not X:
        print("No features extracted. Please check your dataset.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)
    return X, y, label_encoder


def prepare_data_loaders(X, y, batch_size=32, test_size=0.2):
    """
    Split data into training and testing sets, then prepare DataLoaders.

    Parameters:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        batch_size (int): Batch size for DataLoaders.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: Train DataLoader, Test DataLoader.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension for CNN
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
