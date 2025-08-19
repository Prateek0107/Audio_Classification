import streamlit as st
import numpy as np
import librosa
import pickle
import soundfile as sf
import tempfile

# Load the trained model
with open("./assests/audio_classification.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# UrbanSound8K class labels
CLASS_LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark",
    "Drilling", "Engine Idling", "Gunshot", "Jackhammer", "Siren", "Street Music"
]

# Feature extraction function
def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
    mfccs_feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_feature = np.mean(mfccs_feature.T, axis=0)
    return mfccs_scaled_feature

# 🎨 Streamlit UI Design
st.set_page_config(page_title="Audio Classifier", page_icon="🎵", layout="centered")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🎶 Audio Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an audio file (WAV, MP3) to classify it!</p>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("🎤 Upload an Audio File:", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 🎧 Display the uploaded audio
    st.audio(uploaded_file, format="audio/wav")

    # 🎵 Extract features
    features = feature_extractor(tmp_path).reshape(1, -1)

    # 🧠 Make prediction
    predictions = model.predict(features)
    class_index = np.argmax(predictions)
    predicted_class = CLASS_LABELS[class_index]
    confidence = predictions[0][class_index] * 100  # Convert confidence to percentage

    # 🌟 Display Prediction
    st.markdown("<h3 style='text-align: center; color: #27AE60;'>✅ Prediction Result</h3>", unsafe_allow_html=True)
    st.success(f"🎤 **Predicted Class:** {predicted_class}  \n🔍 **Confidence:** {confidence:.2f}%")

    # 🎨 Add a colored bar for confidence score
    st.progress(int(confidence))

# Footer
st.markdown("<br><hr><p style='text-align: center; color: gray;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
