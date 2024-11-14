# feature_extraction.py


import librosa
import numpy as np
import inspect

def extract_features(file_path, fixed_length=100, sr=22050):
    """
    Extracts MFCC features from an audio file.

    Parameters:
        file_path (str): Path to the audio file.
        fixed_length (int): Fixed number of time frames.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Extracted MFCC features of shape (40, 100) or None if extraction fails.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Pad or truncate to fixed_length
        if mfcc.shape[1] < fixed_length:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :fixed_length]
        return mfcc
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def extract_features_from_signal(y, fixed_length=100, sr=22050):
    """
    Extracts MFCC features from an audio signal array.

    Parameters:
        y (np.ndarray): Audio signal.
        fixed_length (int): Fixed number of time frames.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Extracted MFCC features of shape (40, 100) or None if extraction fails.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Pad or truncate to fixed_length
        if mfcc.shape[1] < fixed_length:
            pad_width = fixed_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :fixed_length]
        return mfcc
    except Exception as e:
        print(f"Error extracting features from signal: {e}")
        return None

def add_noise(y, noise_factor=0.005):
    """
    Adds random noise to an audio signal for data augmentation.

    Parameters:
        y (np.ndarray): Original audio signal.
        noise_factor (float): Factor to control noise level.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    noise = np.random.randn(len(y))
    y_noisy = y + noise_factor * noise
    # Cast back to same data type
    y_noisy = y_noisy.astype(type(y[0]))
    return y_noisy

def augment_time_stretch(y, rate=1.05):
    """
    Stretches the audio signal by a given rate for data augmentation.

    Parameters:
        y (np.ndarray): Original audio signal.
        rate (float): Stretch factor.

    Returns:
        np.ndarray: Time-stretched audio signal.
    """
    try:
        y_stretched = librosa.effects.time_stretch(y, rate=rate)  # Keyword argument
        return y_stretched
    except Exception as e:
        print(f"Error in time stretching: {e}")
        return y  # Return original audio if time stretching fails

if __name__ == "__main__":
    # Example usage for testing
    import sys
    if len(sys.argv) != 2:
        print("Usage: python feature_extraction.py <audio_file>")
    else:
        file_path = sys.argv[1]
        features = extract_features(file_path)
        if features is not None:
            print(f"Extracted features shape: {features.shape}")
        else:
            print("Feature extraction failed.")
