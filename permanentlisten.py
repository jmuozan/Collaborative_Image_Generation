from openai import OpenAI
import os
import requests
import sounddevice as sd
import speech_recognition as sr
import threading
from queue import Queue
from scipy.io.wavfile import write
from dotenv import load_dotenv
from collections import deque
import time
import numpy as np

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
HISTORY_FILE = "history.txt"
MAX_HISTORY_LINES = 10
IMAGE_FOLDER = "Generated_Images"
BATCH_SIZE = 3  # Trigger image generation every 3 inputs
DURATION = 5  # Duration of audio capture in seconds
FS = 16000  # Sample rate for Whisper (16kHz is recommended)

# Ensure the image folder exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Queue for transcription and image generation batches
input_queue = Queue()
generation_in_progress = threading.Event()  # Track if generation is in progress

# Function to detect and transcribe speech with Whisper
def capture_audio_input():
    while True:
        print("Listening for input...")
        # Record audio for the specified duration
        audio_data = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        # Check if the audio data is non-empty
        if audio_data is None or len(audio_data) == 0:
            print("No audio data captured.")
            continue
        
        # Save the audio data to a temporary WAV file
        audio_file_path = "temp_audio.wav"
        print(f"Saving audio data to {audio_file_path}...")
        write(audio_file_path, FS, audio_data)
        
        # Confirm that the file was written and has a size
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            print("Error: temp_audio.wav file was not written correctly.")
            continue
        try:
            # Transcribe audio using OpenAI's Whisper API
            with open(audio_file_path, "rb") as audio_file:
                transcription_result = client.audio.transcribe("whisper-1", audio_file)
                transcription = transcription_result.text
                print(f"Recognized input: {transcription}")
                input_queue.put(transcription)  # Add transcription to queue
        except Exception as e:
            print(f"Error with transcription: {e}")

# Function to update and retrieve history
def update_and_get_history(new_inputs):
    history = deque(maxlen=MAX_HISTORY_LINES)
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            for line in file:
                history.append(line.strip())

    history.extend(new_inputs)
    with open(HISTORY_FILE, "a") as file:
        for input in new_inputs:
            file.write(f"{input}\n")

    return list(history)

# Get the next available filename for saving images
def get_next_image_filename():
    existing_files = os.listdir(IMAGE_FOLDER)
    image_numbers = [
        int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith("image_") and f.endswith(".jpeg")
    ]
    next_number = max(image_numbers) + 1 if image_numbers else 1
    return f"image_{next_number}.jpeg"

# Function to generate an image based on user inputs
def generate_image(system_role, user_inputs):
    history_inputs = update_and_get_history(user_inputs)
    combined_history = ". ".join(history_inputs)
    combined_inputs = ". ".join(user_inputs)

    final_prompt = (
        f"{system_role}\n\n"
        f"Here is the collaborative context from multiple users: {combined_inputs}. "
        f"Additionally, here is the recent history of inputs: {combined_history}. "
    )

    try:
        response = client.images.generate(model="dall-e-3", prompt=final_prompt,
        n=1,
        size="1024x1024")

        # Extract the image URL from the response
        image_url = response.data[0].url
        print(f"Image URL: {image_url}")

        img_data = requests.get(image_url).content
        img_filename = get_next_image_filename()
        img_filepath = os.path.join(IMAGE_FOLDER, img_filename)

        with open(img_filepath, 'wb') as handler:
            handler.write(img_data)

        print(f"Image saved as {img_filepath}")
        return img_filepath

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Worker function to monitor input queue and trigger generation
def process_inputs():
    system_role = (
        "You are an AI model specializing in collaborative art generation. "
        "Your role is to combine multiple user inputs in a hand-drawn sketch illustration on a black background, "
        "with a focus on outlines and a childlike, minimalistic but colorful style. "
        "Always use a black background to ensure consistency. "
        "Use a 16:9 aspect ratio to ensure consistency."
    )

    batch = []
    while True:
        transcription = input_queue.get()  # Get transcription from the queue
        batch.append(transcription)

        if len(batch) == BATCH_SIZE:
            # If generation is in progress, wait for it to complete
            if generation_in_progress.is_set():
                print("Generation in progress, waiting to queue next batch.")
                generation_in_progress.wait()  # Wait until the current generation finishes

            # Start image generation with the collected batch
            generation_in_progress.set()  # Mark generation as in progress
            threading.Thread(target=generate_and_clear_batch, args=(system_role, batch)).start()
            batch = []  # Reset batch

# Generate image and clear batch
def generate_and_clear_batch(system_role, batch):
    print("Generating image with batch:", batch)
    generate_image(system_role, batch)
    generation_in_progress.clear()  # Mark generation as complete

# Start the audio capture and processing threads
if __name__ == "__main__":
    # Start listening thread
    threading.Thread(target=capture_audio_input, daemon=True).start()

    # Start processing thread
    threading.Thread(target=process_inputs, daemon=True).start()

    # Keep the main thread alive
    while True:
        time.sleep(1)