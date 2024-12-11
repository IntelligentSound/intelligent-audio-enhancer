import librosa
import numpy as np
from spleeter.separator import Separator
from openl3 import process_file

def separate_audio(file_path, output_dir="separated_audio"):
    """
    Separates audio into ingredients (vocals, drums, bass, etc.) using Spleeter.

    Parameters:
        file_path (str): Path to the audio file.
        output_dir (str): Directory to save separated audio files.

    Returns:
        dict: Paths to separated audio files (vocals, drums, bass, etc.).
    """
    try:
        # Initialize Spleeter with a pre-trained model
        separator = Separator('spleeter:4stems')  # Separate into vocals, drums, bass, and others
        separator.separate_to_file(file_path, output_dir)

        # File paths to separated components
        separated_files = {
            "vocals": f"{output_dir}/{file_path.split('/')[-1].replace('.wav', '')}/vocals.wav",
            "drums": f"{output_dir}/{file_path.split('/')[-1].replace('.wav', '')}/drums.wav",
            "bass": f"{output_dir}/{file_path.split('/')[-1].replace('.wav', '')}/bass.wav",
            "other": f"{output_dir}/{file_path.split('/')[-1].replace('.wav', '')}/other.wav"
        }

        return separated_files
    except Exception as e:
        print(f"Error during audio separation: {e}")
        return None


def extract_features_with_openl3(file_path, sr=22050, fixed_length=100):
    """
    Extracts audio features using OpenL3 embeddings and additional audio features.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        fixed_length (int): Fixed number of time frames.

    Returns:
        np.ndarray: Combined feature array of shape (n_features, fixed_length).
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)

        # OpenL3 Embeddings
        embeddings, _ = process_file(file_path, input_repr="mel256", content_type="music")
        embeddings = embeddings.T  # Transpose for compatibility

        # Additional features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)

        # Standardize lengths by truncating or padding
        def pad_or_truncate(feature):
            if feature.shape[1] < fixed_length:
                pad_width = fixed_length - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :fixed_length]
            return feature

        # Apply padding/truncation
        embeddings = pad_or_truncate(embeddings)
        spectral_bandwidth = pad_or_truncate(spectral_bandwidth)
        rmse = pad_or_truncate(rmse)

        # Combine features into a single array
        combined_features = np.vstack([embeddings, spectral_bandwidth, rmse])

        return combined_features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
