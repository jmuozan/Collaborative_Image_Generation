from openai import OpenAI
import os
import requests
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

##################################
#########   Load Inputs
##################################


# Folder to store generated images
IMAGE_FOLDER = "Generated_Images"


# Ensure the image folder exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

def get_next_image_filename():
    """Find the next available filename in the IMAGE_FOLDER with an incrementing number."""
    existing_files = os.listdir(IMAGE_FOLDER)
    # Extract numbers from filenames like "image_1.jpeg", "image_2.jpeg", etc.
    image_numbers = [
        int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith("image_") and f.endswith(".jpeg")
    ]
    
    # Determine the next available number
    next_number = max(image_numbers) + 1 if image_numbers else 1
    return f"image_{next_number}.jpeg"


# File to store input history
HISTORY_FILE = "history.txt"
MAX_HISTORY_LINES = 3

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


# Function to generate an image based on the combined prompt
def generate_image(system_role, user_inputs ):
    # Combine user inputs into a single descriptive prompt
    history_inputs = update_and_get_history(user_inputs)
    print(history_inputs)
    combined_history = ". ".join(history_inputs)
    combined_inputs = ". ".join(user_inputs)
    final_prompt = (
        f"{system_role}\n\n"
        f"Here is the collaborative context from multiple users: {combined_inputs}. "
        f"Additionally, here is the recent history of inputs: {combined_history}. "
    )

    try:
        # Send the prompt to OpenAI's DALL-E API for image generation
        response = client.images.create_variation(model="dall-e-2",image=open("Generated_Images/image_9.jpeg"), prompt=final_prompt,
        n=1,
        size="1024x1024")

        # Extract the image URL from the response
        image_url = response.data[0].url
        print(f"Image URL: {image_url}")

        # Download and save the image locally with an incrementing filename
        img_data = requests.get(image_url).content
        img_filename = get_next_image_filename()
        img_filepath = os.path.join(IMAGE_FOLDER, img_filename)

        # Download and save the image locally
        img_data = requests.get(image_url).content
        img_filename = "Generated_Image.jpeg"

        with open(img_filepath, 'wb') as handler:
            handler.write(img_data)

        print(f"Image saved as {img_filepath}")
        return img_filepath

    except Exception as e:
        print(f"Error generating image: {e}")
        return None


##################################
#########   Call Functions
##################################


if __name__ == "__main__":
    # Define the system role for the image generation
    system_role = (
        "You are an AI model specializing in collaborative art generation. "
        "You are an AI on a Light outdoor exhibition. "
        #"Your role is to combine multiple user inputs in a hand-drawn sketch illustration on a black background, with a focus on outlines and a childlike, minimalistic but colorful style."
        #"Your role is to combine multiple user inputs in a artistic illustration style on a black background, with a focus on outlines and minimalistic but colorful style."
        #"Your role is to combine multiple user inputs in a artistic illustration style on a black background, with a focus on outlines and minimalistic but strong saturated colorful style, with neon colores."
        "Your role is to combine multiple user inputs in a hand drawn illustration style on a black background, with a focus on outlines and minimalistic but strong saturated colorful style and high contrast colors."
        "use always a black background to ensure consistency"
        "Use a 16:9 aspect ratio to ensure consistency."
    )

    user_inputs = [
        "remove the lights",
        "remove humans",
        "place cars"
    ]

    # Collect user inputs (already captured in the previous steps)
    #user_inputs = []
    #for i in range(1, 4):
      #  user_input = capture_audio_input(i)
       # if user_input:
       #    user_inputs.append(user_input)

    # Generate the image with the system role and combined user inputs
    img_path = generate_image(system_role, user_inputs)
    print("Process complete.")

