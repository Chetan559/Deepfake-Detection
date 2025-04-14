from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import base64
# for audio files
import torch
import torchaudio
import torch.nn.functional as F
import io
from io import BytesIO
import matplotlib.pyplot as plt
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead

app = Flask(__name__)

CORS(app)
model = keras.models.load_model('./Xception_Face2Face.h5')

IMG_SIZE = 299
MAX_SEQ_LENGTH = 30
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = tf.keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.xception.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask
  
# Load audio function
def load_audio(audio_bytes, sampling_rate=22000):
    try:
        # Load the audio from bytes
        audio, lsr = torchaudio.load(io.BytesIO(audio_bytes))

        # If stereo, convert to mono by averaging both channels
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        audio = torch.clip(audio, -1, 1)  # Clip audio between -1 and 1

        # Make sure the shape is [1, time_steps] (mono, unbatched) before adding batch dim
        return audio  # No unsqueeze here, leave it as [channels, time_steps]
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

# Audio classification function
def classify_audio_clip(clip):
    try:
        # Define your classifier (replace it with your own model architecture)
        classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                        resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                        dropout=0, kernel_size=5, distribute_zero_label=False)
        state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
        classifier.load_state_dict(state_dict)

        # If the clip is 2D (e.g. [channels, time_steps]), add a batch dimension: [batch, channels, time_steps]
        if clip.dim() == 2:
            clip = clip.unsqueeze(0)

        results = F.softmax(classifier(clip), dim=-1)
        return results[0][0]
    except Exception as e:
        print(f"Error in classification: {e}")
        raise


# Function to generate waveform plot using matplotlib
def generate_waveform_plot(audio_clip):
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(audio_clip.squeeze().numpy())
        plt.title("Waveform Plot")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # Save the plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Encode the bytes as base64
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Return the base64-encoded image
        return encoded_image
    except Exception as e:
        print(f"Error generating waveform plot: {e}")
        raise
  
@app.route('/')
def index():
  html ='''<h1> hello this page is working</h1>
            <h2> send post request to /predict_video with video file</h2>
            <h2> the response will be json with video, result and confidence</h2>
            <h2> the video will be base64 encoded</h2>
            <h2> send post request to /predict_audio with audio file</h2>
            <h2> the response will be json with audio, result and waveform image</h2>'''
  return html


@app.route('/predict_video', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    # Ensure the 'upload' directory exists
    if not os.path.exists('upload'):
        os.makedirs('upload')

    video_path = os.path.join("upload", video.filename)

    try:
        video.save(video_path)
    except Exception as e:
        return jsonify({'error': 'Failed to save video file', 'details': str(e)}), 500

    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)

    prediction = model.predict([frame_features, frame_mask])[0]
    
    if prediction > 0.451:
      result = 'FAKE'

    else:
      result = 'REAL'

    # prediction = prediction*100
    if prediction > 0.451:
        confidence = float(prediction)
    else:
        confidence = 1 - float(prediction)
        
    with open(video_path, "rb") as video_file:
        video_encoded = base64.b64encode(video_file.read()).decode('utf-8')

    return jsonify({'video': video_encoded, 'result': result, 'confidence': confidence})
  



@app.route('/predict_audio', methods=['POST'])
def upload_audio():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Process the audio file here
            audio_bytes = file.read()

            # Load and classify the audio file
            audio_clip = load_audio(audio_bytes)
            result = classify_audio_clip(audio_clip)

            # Generate waveform plot
            waveform_image = generate_waveform_plot(audio_clip)

            # Convert audio file to base64 for returning in JSON
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_mime = f"data:audio/mp3;base64,{audio_b64}"

            # Return a JSON response containing result, audio, and waveform image
            return jsonify({
                'result': result.item(),
                'audio': audio_mime,
                'waveform_image': waveform_image
            }), 200

        return jsonify({'error': 'File processing failed'}), 500

    except Exception as e:
        print(f"Error in processing the request: {e}")
        return jsonify({'error': str(e)}), 500


# Start the Flask app
if __name__ == '__main__':
  app.run()