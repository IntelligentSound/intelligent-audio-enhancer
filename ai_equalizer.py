import argparse
from feature_extraction import extract_advanced_features
from data_preparation import load_data_with_separation
from genre_classifier import train_and_evaluate_model
from sound_optimization import prepare_adjustment_data, train_adjustment_model, enhance_audio

def main():
    parser = argparse.ArgumentParser(description="AI Equalizer for enhancing audio dynamically")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the genre classifier and adjustment models")
    group.add_argument("--enhance", type=str, help="Path to the audio file to enhance")
    parser.add_argument("--output", type=str, help="Path to save the enhanced audio file", default="output_enhanced.wav")
    parser.add_argument("--dataset", type=str, help="Path to the dataset directory for training", default="audio_samples/")
    args = parser.parse_args()

    if args.train:
        print("Starting training...")
        audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
        genres = ["Genre1", "Genre2"]
        X, y, label_encoder = load_data_with_separation(audio_files, genres, use_separation=True)
        train_loader, test_loader = prepare_data_loaders(X, y)
        model = train_and_evaluate_model(train_loader, test_loader, num_classes=len(set(genres)), n_features=100, fixed_length=100)
        adjustments = [[+3, -2, +1], [-1, +2, -1]]
        X_train, X_test, y_train, y_test, scaler = prepare_adjustment_data(X, adjustments)
        adjustment_model = train_adjustment_model(X_train, y_train)
        print("Training completed. Models are ready to use.")
    elif args.enhance:
        audio_path = args.enhance
        output_path = args.output
        adjustment_model = ...  # Load your adjustment model here
        scaler = ...  # Load the scaler used during training
        enhance_audio(audio_path, extract_advanced_features, adjustment_model, scaler)
        print(f"Enhanced audio saved to: {output_path}")

if __name__ == "__main__":
    main()
