from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import numpy as np

def prepare_adjustment_data(features, adjustments, test_size=0.2):
    """
    Prepares data for training adjustment models by scaling features.

    Parameters:
        features (np.ndarray): Extracted audio features.
        adjustments (list): Desired adjustments for audio components.
        test_size (float): Proportion of data for testing.

    Returns:
        tuple: Scaled training and testing datasets, and the scaler.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, adjustments, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


def train_adjustment_model(X_train, y_train):
    """
    Train a MultiOutputRegressor model for audio adjustments.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training adjustments.

    Returns:
        model: Trained adjustment model.
    """
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, y_train)
    return model


def train_genre_specific_models(features, adjustments, genres):
    """
    Train separate adjustment models for each genre.

    Parameters:
        features (np.ndarray): Extracted audio features.
        adjustments (list): Desired adjustments for audio components.
        genres (list): Genre labels corresponding to features.

    Returns:
        dict: Genre-specific adjustment models.
    """
    genre_models = {}
    unique_genres = np.unique(genres)

    for genre in unique_genres:
        genre_indices = [i for i, g in enumerate(genres) if g == genre]
        X_genre = features[genre_indices]
        y_genre = adjustments[genre_indices]
        X_train, X_test, y_train, y_test, scaler = prepare_adjustment_data(X_genre, y_genre)
        model = train_adjustment_model(X_train, y_train)
        genre_models[genre] = {
            "model": model,
            "scaler": scaler
        }
        print(f"Trained adjustment model for genre: {genre}")

    return genre_models


def enhance_audio(audio_path, feature_extractor, adjustment_model, scaler):
    """
    Enhance audio based on a trained adjustment model and extracted features.

    Parameters:
        audio_path (str): Path to the input audio file.
        feature_extractor (function): Function to extract features from audio.
        adjustment_model (model): Trained adjustment model.
        scaler (StandardScaler): Scaler used for feature normalization.
    """
    try:
        features = feature_extractor(audio_path)
        if features is None:
            print(f"Could not extract features from {audio_path}")
            return

        # Normalize features
        features_scaled = scaler.transform([features])
        adjustments = adjustment_model.predict(features_scaled)

        # Apply adjustments
        audio = AudioSegment.from_file(audio_path)
        if "bass" in adjustments:
            audio = audio.low_pass_filter(150).apply_gain(adjustments[0][0])
        if "treble" in adjustments:
            audio = audio.high_pass_filter(5000).apply_gain(adjustments[0][1])
        if "vocals" in adjustments:
            audio = audio.apply_gain(adjustments[0][2])

        output_path = audio_path.replace(".wav", "_enhanced.wav")
        audio.export(output_path, format="wav")
        print(f"Enhanced audio saved to: {output_path}")
    except Exception as e:
        print(f"Error enhancing audio: {e}")
