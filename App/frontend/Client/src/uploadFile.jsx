import { useState, useRef } from 'react';
import './uploadFile.css';
import axios from 'axios';
import be_url from '../beUrl.js'; // Your video detection backend URL
import audio_be_url from '../audioBeUrl.js'; // Your audio detection backend URL

function UploadFile() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [useAudio, setUseAudio] = useState(false); // State to track checkbox
  const [uploadStatus, setUploadStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [videoSrc, setVideoSrc] = useState(null); // To store the video source
  const fileInput = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setUploadStatus("");
    setVideoSrc(null); // Reset the video source when a new file is selected

    if (file) {
      handleUpload(file); // Automatically upload the file when selected
    }
  };

  const handleUpload = (file) => {
    if (isUploading) return;

    if (file) {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('video', file); // Attach the video file
      formData.append('useAudio', useAudio); // Pass the useAudio flag

      // If 'useAudio' is true, send requests to both video and audio servers
      const videoRequest = axios.post(`${be_url}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true,
      });

      const audioRequest = useAudio
        ? axios.post(`${audio_be_url}/predict`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          withCredentials: true,
        })
        : null;

      // Use Promise.all to wait for both requests if 'useAudio' is true
      Promise.all([videoRequest, audioRequest])
        .then((responses) => {
          const videoResponse = responses[0].data;
          const { video, result: videoResult, confidence: videoConfidence } = videoResponse;

          // Assuming video is a base64 string
          setVideoSrc(`data:video/mp4;base64,${video}`); // Set the video source to be played
          let status = `Video Result: ${videoResult}, Confidence: ${videoConfidence.toFixed(2)}`;

          // If audio is used, add its result
          if (useAudio && responses[1]) {
            const audioResponse = responses[1].data;
            const { result: audioResult, confidence: audioConfidence } = audioResponse;
            status += ` | Audio Result: ${audioResult}, Confidence: ${audioConfidence.toFixed(2)}`;
          }

          setUploadStatus(status);
        })
        .catch((err) => {
          console.error("Upload error:", err);
          setUploadStatus("Failed to upload file.");
        })
        .finally(() => {
          setIsUploading(false);
        });
    } else {
      console.log("No file selected");
      setUploadStatus("No file selected.");
    }
  };

  return (
    <div id='upload-file-wrap'>
      <div className="upload-file">
        <div
          id="upload-file-area"
          onClick={() => fileInput.current.click()}
        >
          <span style={{ color: "white" }}>Click to Upload Video</span>
          <input
            type='file'
            ref={fileInput}
            id='file-input'
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        <div>
          <input
            type="checkbox"
            id="use-audio-checkbox"
            checked={useAudio}
            onChange={() => setUseAudio(!useAudio)} // Toggle the checkbox
          />
          <label htmlFor="use-audio-checkbox">Use Audio</label>
        </div>

        <button
          id='upload-button'
          onClick={() => handleUpload(selectedFile)}
          disabled={!selectedFile || isUploading}
        >
          {isUploading ? "Uploading..." : "Upload Video"}
        </button>

        {uploadStatus && <div id="upload-status">{uploadStatus}</div>}

        {/* Display the video if it exists */}
        {videoSrc && (
          <video
            id="uploaded-video"
            controls
            src={videoSrc}
            height='300'
            width="400"
          />
        )}
      </div>
    </div>
  );
}

export default UploadFile;
