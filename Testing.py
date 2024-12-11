from feature_extraction import extract_advanced_features
from feature_extraction import separate_audio

#Audio seperation
separated_files = separate_audio("Users/bomah/Documents/Ai_equalizer/old song/pop.00001.au")
print("Separated Audio Files:", separated_files)

#Feature extraction
features = extract_advanced_features("/Users/bomah/Documents/Ai_equalizer/old song/pop.00001.au")
print("Extracted Features Shape:", features.shape)