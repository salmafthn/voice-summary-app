from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from speech_to_text import transcribe_audio
import os

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400

    audio_file = request.files['audio']
    
    upload_folder = os.path.join('static', 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, audio_file.filename)
    
    audio_file.save(file_path)

    try:
        transcript = transcribe_audio(file_path)
        return jsonify({'transcript': transcript}), 200  
    except Exception as e:
        return jsonify({'error': str(e)}), 500  

if __name__ == "__main__":
    app.run(debug=True)
