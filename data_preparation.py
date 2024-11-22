from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data_with_separation(audio_files, genres, fixed_length=100, use_separation=True):
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
                    features = extract_advanced_features(ingredient_path, fixed_length=fixed_length)
                    if features is not None:
                        X.append(features)
                        y.append(label)
            else:
                features = extract_advanced_features(file_path, fixed_length=fixed_length)
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
