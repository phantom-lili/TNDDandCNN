import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical

# Audio file paths and corresponding labels
AUDIO_FILES = [
    r"D:\es\jian\50629-4-0-1.wav", r"D:\es\jian\50629-4-1-0.wav", r"D:\es\jian\50629-4-1-1.wav",
    r"D:\es\jian\50629-4-1-2.wav", r"D:\es\jian\50629-4-1-3.wav", r"D:\es\jian\50629-4-1-4.wav",
    r"D:\es\jian\50629-4-1-8.wav", r"D:\es\jian\50629-4-1-9.wav", r"D:\es\jian\50629-4-1-11.wav",
    r"D:\es\jian\50629-4-1-13.wav", r"D:\es\jian\50629-4-3-0.wav", r"D:\es\yu\15544-5-0-0.wav",
    r"D:\es\yu\15544-5-0-1.wav", r"D:\es\yu\15544-5-0-2.wav", r"D:\es\yu\15544-5-0-3.wav",
    r"D:\es\yu\15544-5-0-4.wav", r"D:\es\yu\15544-5-0-5.wav", r"D:\es\yu\15544-5-0-6.wav"
]
LABELS = [0] * 11 + [1] * 7  # Example label list

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512, max_time_steps=100):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    if y.size == 0:
        raise ValueError(f"Failed to load audio file: {file_path}")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs[:, :max_time_steps]  # Truncate MFCC sequences
    return mfccs

def prepare_data(audio_files, labels):
    """Prepare features and labels for training."""
    features = np.array([extract_features(file) for file in audio_files])
    
    # Validate features and labels length
    if len(features) != len(labels):
        raise ValueError("The number of features and labels must be the same.")
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(labels))
    labels_one_hot = to_categorical(labels, num_classes=num_classes)
    
    return features, labels_one_hot, num_classes

def build_model(input_shape, num_classes):
    """Build and compile the CNN model."""
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add Dropout layer to prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, num_classes):
    """Train and evaluate the model."""
    model = build_model(input_shape=(13, 100), num_classes=num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return model

def predict_new_audio(file_path, model):
    """Predict the label of a new audio file."""
    feature = extract_features(file_path)
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    prediction = model.predict(feature)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Main execution
if __name__ == "__main__":
    # Prepare data
    features, labels_one_hot, num_classes = prepare_data(AUDIO_FILES, LABELS)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.2, random_state=42)
    
    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test, num_classes)
    
    # Save the model
    model.save("speaker_recognition_model.h5")

    # Load the model
    loaded_model = load_model("speaker_recognition_model.h5")

    # Test a new audio file
    new_audio_file = r"D:\es\yu\15544-5-0-13.wav"
    predicted_label = predict_new_audio(new_audio_file, loaded_model)
    print(f"Predicted speaker label: {predicted_label}")
