from openai import OpenAI
import os
from PIL import Image
from dotenv import load_dotenv
import requests  # Import requests to handle HTTP requests for image download
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from collections import deque

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Base configuration
IMAGE_SIZE = "1024x1024"
IMAGE_FOLDER = "Generated_Images"
BATCH_SIZE = 3  # Trigger image generation every 3 inputs
# File to store input history
HISTORY_FILE = "history.txt"
MAX_HISTORY_LINES = 7


# Initialize speech recognizer
recognizer = sr.Recognizer()


# Base prompt to add context for DALL-E image generation
base_prompt = (
    "This is a collaborative image based on inputs from multiple users. "
    "The image should creatively represent the combination of all inputs."
)

##################################
#########   Functions
##################################



# Function to capture audio using sounddevice and save to WAV
def capture_audio_input(person_number, filename="user_input.wav", duration=5, fs=44100):
    print(f"Person {person_number}, please say something:")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, (audio_data * 32767).astype('int16'))  # Save as WAV

    # Transcribe the saved audio file
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    try:
        transcription = recognizer.recognize_google(audio)
        print(f"Person {person_number}'s input: {transcription}")
        return transcription
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# Function to update and retrieve history
def update_and_get_history(new_inputs):
    # Read existing history from file
    history = deque(maxlen=MAX_HISTORY_LINES)
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            for line in file:
                history.append(line.strip())

    # Add new inputs to history and write to file
    history.extend(new_inputs)
    with open(HISTORY_FILE, "a") as file:
        for input in new_inputs:
            file.write(f"{input}\n")

    # Return the last 10 inputs as a list for prompt generation
    return list(history)

def generate_image_with_history(system_prompt, history, latest_image_path):
    # Create a combined prompt based on the system prompt, user history, and the latest inputs
    # Combine user inputs into a single descriptive prompt
    history_inputs = update_and_get_history(user_inputs)
    print(history_inputs)
    combined_history = ". ".join(history_inputs)
    combined_inputs = ". ".join(user_inputs)
    combined_prompt = (
        f"{system_role}\n\n"
        f"Modify the image by adding the following new element(s): {combined_inputs}. "
        f"Current context based on previous inputs: {combined_history}. "
        f" Make sure the new element blends seamlessly with the existing elements."
    )
    # If there is an existing image, we use it for inpainting
    if latest_image_path:
        # Ensure the image is in RGBA format
        with Image.open(latest_image_path) as img:
            img = img.convert("RGBA")
            rgba_image_path = "temp_rgba_image.png"
            img.save(rgba_image_path, format="PNG")  # Save as PNG with RGBA

        with open(rgba_image_path, "rb") as img_file:
            # Call the OpenAI API with inpainting options
            response = client.images.edit(
                image=img_file,
                prompt=combined_prompt,
                n=1,
                size=IMAGE_SIZE
            )
    else:
        # No previous image, start fresh
        response = client.images.create(
            prompt=combined_prompt,
            n=1,
            size=IMAGE_SIZE
        )
    
    # Extract the image URL from the response and save it
    image_url = response.data[0].url
    img_filename = get_next_image_filename()
    download_image(image_url, img_filename)
    
    return img_filename

def get_next_image_filename():
    # Generate next image filename, e.g., image_1.jpeg, image_2.jpeg, etc.
    existing_files = os.listdir(IMAGE_FOLDER)
    image_numbers = [
        int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith("image_") and f.endswith(".jpeg")
    ]
    next_number = max(image_numbers) + 1 if image_numbers else 1
    return f"{IMAGE_FOLDER}/image_{next_number}.jpeg"

def download_image(image_url, filename):
    # Download and save the image from the URL
    img_data = requests.get(image_url).content
    with open(filename, 'wb') as handler:
        handler.write(img_data)
    print(f"Image saved as {filename}")

if __name__ == "__main__":
    # Define the system role for the image generation
    system_role = (
        "You are an AI model specializing in collaborative art generation. "
        "You are an AI model generating new picture based on the previous one"
        "You are an AI on a Light outdoor exhibition. "
        #"Your role is to combine multiple user inputs in a hand-drawn sketch illustration on a black background, with a focus on outlines and a childlike, minimalistic but colorful style."
        #"Your role is to combine multiple user inputs in a artistic illustration style on a black background, with a focus on outlines and minimalistic but colorful style."
        #"Your role is to combine multiple user inputs in a artistic illustration style on a black background, with a focus on outlines and minimalistic but strong saturated colorful style, with neon colores."
        "Your role is to combine multiple user inputs in a hand drawn sketch illustration style on a black background, with a focus on outlines and minimalistic but strong saturated colorful style using neon colors."
        "use always a black background to ensure consistency"
        "Use a 16:9 aspect ratio to ensure consistency."
    )

    user_inputs = [
        "remove the lights",
        "remove humans",
        "place cars"
    ]

    # Example user history and latest image file
    user_history = ["we're outdoor", "there is a lot of green", "will ai take over ?"]
    latest_image_path = "Generated_Images/image_9.jpeg"  # Path to the latest image
    
    # Generate the next image in sequence
    new_image_path = generate_image_with_history(system_role, user_inputs, latest_image_path)
    print("New image generated:", new_image_path)
