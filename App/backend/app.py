from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model('./Xception_Face2Face.h5', compile=False)


IMG_SIZE = 299
MAX_SEQ_LENGTH = 30 # or 40
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.xception.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

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
  y,x = frame.shape[0:2]
  min_dim = min(y,x)
  start_x = (x//2) - (min_dim//2)
  start_y = (y//2) - (min_dim//2)
  return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def prepare_single_video(frames):
  frames = frames[None, ...]
  frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
  frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="bool")

  for i, batch in enumerate(frames):
    video_length = batch.shape[0]
    length = min(MAX_SEQ_LENGTH, video_length)
    for j in range(length):

      frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
    frame_mask[i, :length] = 1 #1 = not masked , 0= masked

  return frame_features, frame_mask

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    video_path = os.path.json("upload", video.filename)
    video.save(video_path)
    
    frames = load_video(video_path)
    frames_features, frame_mask = prepare_single_video(frames)
    
    prediction = model.predict([frames_features, frame_mask])[0]
    result = 'FAKE' if prediction >= 0.51 else 'REAL'
    confidence = float(prediction)
    
    os.remove(video_path)
    
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
    