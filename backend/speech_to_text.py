import whisper
import os

# Load model Whisper
model = whisper.load_model("base")  # Anda bisa memilih model yang lebih besar jika diperlukan (e.g. "large")

def transcribe_audio(audio_file_path):
    """
    Fungsi untuk mengonversi audio menjadi teks menggunakan Whisper API.
    audio_file_path: Lokasi file audio yang diunggah.
    """
    # Transkripsikan audio menggunakan Whisper
    result = model.transcribe(audio_file_path)
    return result["text"]
