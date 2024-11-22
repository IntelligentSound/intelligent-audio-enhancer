from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

def prepare_adjustment_data(features, adjustments):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, adjustments, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def train_adjustment_model(X_train, y_train):
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, y_train)
    return model

def apply_dynamic_adjustments(audio_path, adjustments):
    audio = AudioSegment.from_file(audio_path)
    if "bass" in adjustments:
        audio = audio.low_pass_filter(150).apply_gain(adjustments["bass"])
    if "treble" in adjustments:
        audio = audio.high_pass_filter(5000).apply_gain(adjustments["treble"])
    if "vocals" in adjustments:
        audio = audio.apply_gain(adjustments["vocals"])
    return audio
