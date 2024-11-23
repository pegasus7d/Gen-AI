from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the Faster Whisper model
model = WhisperModel("base", device="cpu")  # Use GPU with `device="cuda"` if available

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Retrieve the audio file from the request
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Save the audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            temp_audio_path = temp_file.name

        # Transcribe audio using Faster Whisper
        segments, _ = model.transcribe(temp_audio_path)

        # Combine segments into a full transcription
        transcription = " ".join([segment.text for segment in segments])

        # Cleanup the temporary file
        os.remove(temp_audio_path)

        # Return the transcription
        return jsonify({"transcription": transcription})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
