from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

Xception_url = "https://4456-34-73-111-58.ngrok-free.app/predict"  # Update your video model endpoint
Audio_url = "https://af66-34-86-98-168.ngrok-free.app/predict"  # Update audio model endpoint

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video = request.files['video']
    use_audio = request.form.get('useAudio') == 'true'  # Handle useAudio flag

    try:
        # Send request to video model server
        response_xception = requests.post(Xception_url, files={'video': video})
        if response_xception.status_code != 200:
            return jsonify({"error": "Video model failed"}), 500

        video_result = response_xception.json()
        result = {"video_result": video_result}
        
        # If 'useAudio' is selected, request audio model as well
        if use_audio:
            response_audio = requests.post(Audio_url, files={'audio': video})  # Extract audio
            if response_audio.status_code != 200:
                return jsonify({"error": "Audio model failed"}), 500
            
            audio_result = response_audio.json()
            result["audio_result"] = audio_result
        
        # Return both video and audio results if audio is selected
        return jsonify(result)
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request to model failed: {e}")
        return jsonify({"error": "Model request failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
