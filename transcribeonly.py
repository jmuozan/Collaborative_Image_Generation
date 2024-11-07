import whisper
import torch
from silero_vad import get_speech_timestamps, read_audio
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load Whisper model
model = whisper.load_model("base")  # Choose the model size as per your requirements

# Load Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github', trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

# Sampling rate for recording
fs = 16000

# Function to continuously record audio in chunks
def record_audio_chunk(duration, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    return audio.flatten()

# Main function to capture live audio, detect speech, and transcribe
def live_transcribe():
    print("Starting live transcription. Press Ctrl+C to stop.")
    try:
        while True:
            # Record a chunk of audio
            audio_chunk = record_audio_chunk(duration=5, fs=fs)  # Record for 5 seconds at a time
            
            # Convert recorded audio chunk to a format Silero VAD can use
            audio_data = np.int16(audio_chunk)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(audio_data, vad_model, sampling_rate=fs)
            
            # Process each detected speech segment
            for idx, ts in enumerate(speech_timestamps):
                # Extract the speech segment from the recorded chunk
                start, end = ts['start'], ts['end']
                speech_segment = audio_data[start:end]
                
                # Save speech segment to a temporary file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                    wav.write(tmp_file.name, fs, speech_segment)
                    
                    # Transcribe using Whisper
                    result = model.transcribe(tmp_file.name)
                    print(f"Transcription {idx + 1}: {result['text']}")
                    
    except KeyboardInterrupt:
        print("Transcription stopped.")

# Start live transcription
live_transcribe()
