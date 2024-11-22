import librosa
import numpy as np
from spleeter.separator import Separator

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


def extract_advanced_features(file_path, sr=22050, fixed_length=100):
    """
    Extracts advanced audio features including chromagram, spectral contrast, zero-crossing rate, 
    tonal centroid (tonnetz), and onset strength.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        fixed_length (int): Fixed number of time frames.

    Returns:
        np.ndarray: Combined feature array of shape (n_features, fixed_length).
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)

        # Extract individual features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        # Standardize lengths by truncating or padding
        def pad_or_truncate(feature):
            if feature.shape[1] < fixed_length:
                pad_width = fixed_length - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :fixed_length]
            return feature

        # Apply padding/truncation
        mfcc = pad_or_truncate(mfcc)
        chroma = pad_or_truncate(chroma)
        spectral_contrast = pad_or_truncate(spectral_contrast)
        zero_crossing_rate = pad_or_truncate(zero_crossing_rate)
        onset_strength = np.pad(onset_strength, (0, max(0, fixed_length - len(onset_strength))), mode='constant')[:fixed_length]
        tonnetz = pad_or_truncate(tonnetz)

        # Combine features into a single array
        combined_features = np.vstack([
            mfcc, chroma, spectral_contrast, zero_crossing_rate, tonnetz, onset_strength[np.newaxis, :]
        ])

        return combined_features
    except Exception as e:
        print(f"Error extracting advanced features from {file_path}: {e}")
        return None
