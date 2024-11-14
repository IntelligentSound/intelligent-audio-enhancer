# ai_equalizer.py


import argparse
from pydub import AudioSegment
import torch
import os
import joblib
import logging
from feature_extraction import extract_features
from model import EnhancedGenreClassifier  # Import from model.py
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


# Configure logging
logging.basicConfig(
    filename='ai_equalizer.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_model(model_path, num_classes, n_features, fixed_length):
    """
    Load the trained genre classifier model from a file.

    Parameters:
        model_path (str): Path to the model file.
        num_classes (int): Number of genre classes.
        n_features (int): Number of MFCC features.
        fixed_length (int): Number of time frames used in feature extraction.

    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
        device (torch.device): Computation device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedGenreClassifier(num_classes, n_features, fixed_length).to(device)
    try:
        # Suppress FutureWarning by ignoring it or addressing it as per PyTorch version
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise e
    return model, device

def predict_genre(model, label_encoder, file_path, device, fixed_length=100):
    """
    Predicts the genre of an audio file using the loaded model.

    Parameters:
        model (torch.nn.Module): Trained genre classifier model.
        label_encoder (LabelEncoder): Encoder for genre labels.
        file_path (str): Path to the audio file.
        device (torch.device): Computation device.
        fixed_length (int): Fixed number of time frames for feature extraction.

    Returns:
        str: Predicted genre label or None if prediction fails.
    """
    features = extract_features(file_path, fixed_length=fixed_length)
    if features is None:
        logging.error(f"Feature extraction failed for {file_path}")
        return None

    # Convert to tensor and reshape
    features_tensor = torch.tensor(features).float().to(device)
    features_tensor = features_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 40, 100]

    with torch.no_grad():
        output = model(features_tensor)
        _, predicted = torch.max(output, 1)
    genre = label_encoder.inverse_transform([predicted.item()])[0]
    return genre

def apply_equalizer_settings(audio, genre):
    """
    Apply predefined equalizer settings based on genre to enhance audio.

    Parameters:
        audio (AudioSegment): Loaded audio file.
        genre (str): Predicted genre of the audio.

    Returns:
        AudioSegment: Enhanced audio with genre-based equalizer applied.
    """
    # Define genre-based frequency adjustments
    if genre == 'Hip-Hop':
        # Boost low frequencies for bass-heavy sound
        audio = audio.low_pass_filter(150).apply_gain(+5)  # Boost low end
        audio = audio.high_pass_filter(5000).apply_gain(-2)  # Slightly reduce high end
    elif genre == 'Jazz':
        # Enhance mids and highs for a clean, smooth sound
        audio = audio.high_pass_filter(500).apply_gain(+3)  # Boost mids and highs
    elif genre == 'Classical':
        # Preserve clarity across the full range without excessive boosting
        audio = audio.high_pass_filter(100).apply_gain(+2)
        audio = audio.low_pass_filter(10000).apply_gain(+1)
    elif genre == 'Rock':
        # Emphasize mids and highs for clarity in vocals and instruments
        audio = audio.high_pass_filter(200).apply_gain(+3)
        audio = audio.low_pass_filter(8000).apply_gain(-1)
    else:
        # Default equalization for other genres
        audio = audio.high_pass_filter(100).apply_gain(+1)
        audio = audio.low_pass_filter(10000).apply_gain(+1)
    # Additional genre-based adjustments as needed...

    return audio

def enhance_batch(input_dir, output_dir, model, label_encoder, device, fixed_length=100):
    """
    Enhances all audio files in the input directory and saves them to the output directory.

    Parameters:
        input_dir (str): Directory containing input audio files.
        output_dir (str): Directory to save enhanced audio files.
        model (torch.nn.Module): Loaded model.
        label_encoder (LabelEncoder): Label encoder.
        device (torch.device): Computation device.
        fixed_length (int): Fixed number of time frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_name.lower().endswith(('.wav', '.mp3', '.au', '.flac')):
            try:
                print(f"Processing {file_name}...")
                genre = predict_genre(model, label_encoder, file_path, device, fixed_length=fixed_length)
                if genre is None:
                    print(f"Failed to predict genre for {file_name}. Skipping...")
                    continue
                audio = AudioSegment.from_file(file_path)
                enhanced_audio = apply_equalizer_settings(audio, genre)
                output_file_path = os.path.join(output_dir, f"enhanced_{file_name}")
                output_format = os.path.splitext(output_file_path)[1][1:].lower() or 'mp3'
                enhanced_audio.export(output_file_path, format=output_format)
                print(f"Saved enhanced audio to {output_file_path}")
                logging.info(f"Enhanced audio saved to {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")
                print(f"Failed to process {file_name}: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AI-based Audio Equalizer')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_file', type=str, help='Path to the input audio file')
    group.add_argument('--input_dir', type=str, help='Path to the input directory containing audio files')
    parser.add_argument('--output_file', type=str, help='Path to save the enhanced audio file')
    parser.add_argument('--output_dir', type=str, help='Path to save enhanced audio files')
    parser.add_argument('--model_path', type=str, default='enhanced_genre_classifier.pth', help='Path to the trained model file')
    parser.add_argument('--label_encoder_path', type=str, default='label_encoder.joblib', help='Path to the label encoder file')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of genre classes in the model')
    parser.add_argument('--fixed_length', type=int, default=100, help='Fixed number of time frames for feature extraction')

    args = parser.parse_args()

    # Load the label encoder
    try:
        label_encoder = joblib.load(args.label_encoder_path)
        logging.info(f"Label encoder loaded from {args.label_encoder_path}")
    except Exception as e:
        logging.error(f"Failed to load label encoder from {args.label_encoder_path}: {e}")
        print(f"Failed to load label encoder from {args.label_encoder_path}: {e}")
        return

    # Load the genre classifier model
    try:
        n_features = 40  # Number of MFCCs used during training
        model, device = load_model(args.model_path, args.num_classes, n_features, args.fixed_length)
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    if args.input_file and args.output_file:
        # Single file processing
        # Load the audio file
        try:
            audio = AudioSegment.from_file(args.input_file)
        except Exception as e:
            logging.error(f"Failed to load audio file {args.input_file}: {e}")
            print(f"Failed to load audio file {args.input_file}: {e}")
            return

        # Predict genre
        genre = predict_genre(model, label_encoder, args.input_file, device, fixed_length=args.fixed_length)
        if genre is None:
            print("Genre prediction failed.")
            return
        print(f"Predicted Genre: {genre}")
        logging.info(f"Predicted Genre for {args.input_file}: {genre}")

        # Apply genre-specific equalization
        enhanced_audio = apply_equalizer_settings(audio, genre)

        # Export enhanced audio to specified path
        try:
            # Determine output format from file extension
            output_format = os.path.splitext(args.output_file)[1][1:].lower() or 'mp3'
            enhanced_audio.export(args.output_file, format=output_format)
            print(f"Enhanced audio saved to {args.output_file}")
            logging.info(f"Enhanced audio saved to {args.output_file}")
        except Exception as e:
            logging.error(f"Failed to save enhanced audio to {args.output_file}: {e}")
            print(f"Failed to save enhanced audio to {args.output_file}: {e}")
    elif args.input_dir and args.output_dir:
        # Batch processing
        try:
            enhance_batch(args.input_dir, args.output_dir, model, label_encoder, device, fixed_length=args.fixed_length)
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            print(f"Batch processing failed: {e}")
    else:
        print("Invalid arguments. Provide either input_file and output_file or input_dir and output_dir.")


if __name__ == "__main__":
    main()
