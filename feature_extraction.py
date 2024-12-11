import librosa
import torchaudio
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
        separator = Separator('spleeter:4stems')  # Separate into vocals, drums, bass, and others
        separator.separate_to_file(file_path, output_dir)

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


def extract_features_with_torchaudio(file_path, sr=22050, fixed_length=100):
    """
    Extracts audio features using torchaudio and additional features.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        fixed_length (int): Fixed number of time frames.

    Returns:
        np.ndarray: Combined feature array of shape (n_features, fixed_length).
    """
    try:
        waveform, sr = torchaudio.load(file_path)

        # Resample if necessary
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=22050)(waveform).numpy()

        # Additional features using librosa
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram[0]), sr=22050, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(S=librosa.power_to_db(mel_spectrogram[0]), sr=22050)

        # Pad or truncate to fixed_length
        def pad_or_truncate(feature):
            if feature.shape[1] < fixed_length:
                pad_width = fixed_length - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :fixed_length]
            return feature

        mel_spectrogram = pad_or_truncate(mel_spectrogram[0])
        mfcc = pad_or_truncate(mfcc)
        chroma = pad_or_truncate(chroma)

        # Combine features into a single array
        combined_features = np.vstack([mel_spectrogram, mfcc, chroma])

        return combined_features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
