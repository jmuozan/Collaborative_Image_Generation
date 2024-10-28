import speech_recognition as sr
from openai import OpenAI
import requests
import os
import time

# Initialize speech recognizer and OpenAI API
recognizer = sr.Recognizer()
openai_api_key = ""  # Replace with your OpenAI API Key
openAI_client = OpenAI(api_key=openai_api_key)
''''''
# Record audio and convert to text (simulated for testing purposes)
with sr.Microphone() as source:
    print("Say something!")
    audio = recognizer.listen(source, 2, 4)  # 3: Seconds to start detecting, 4: Total seconds of recording.
audio_transcription = recognizer.recognize_google(audio)


# Testing without the mic
# audio_transcription = "ADD MORE TREES!"

print(f"Transcription: {audio_transcription}")
# Additional OpenAI prompt modifications if needed
Open_AI_prompt = "This is a collaborative visualization tool where each user's input builds on the previous one. The GPT will generate a simple, hand-drawn sketch illustration on a black background, with a focus on outlines and a childlike, minimalistic style. The output of the previous person's input will serve as a reference for the next user's image, gradually adding more detail with each step. The aim is to create evolving, playful visuals, driven by user contributions, resembling simple sketches that build up over time in complexity. All sketches will always use a black background to maintain consistency. The GPT will always directly output a generated image in a 16:9 format without further asking, once it receives a request."
complete_prompt = audio_transcription + " with this idea " + Open_AI_prompt

# Function to ensure IMG folder exists
def create_img_folder():
    folder_path = 'IMG'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder {folder_path} created.')
    else:
        print(f'Folder {folder_path} already exists.')
    return folder_path

# Generate a filename to save
def generate_filename(prefix="Generated_Image"):
    current_time = time.strftime("%H-%M-%d")  
    filename = f"{prefix}_{current_time}.jpeg"
    return filename

# Generate an image using OpenAI
def generate_image(prompt):
    print(f"Generating image from prompt: {prompt}")
    response = openAI_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    print(f"Image URL: {image_url}")
    img_data = requests.get(image_url).content

    folder_path = create_img_folder()
    img_filename = generate_filename()
    img_file_path = os.path.join(folder_path, img_filename)

    # Save image
    with open(img_file_path, 'wb') as handler:
        handler.write(img_data)

    print(f"Image saved to {img_file_path}")
    return img_file_path

def get_last_image_path(folder_path):
    images = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return None
    
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return os.path.join(folder_path, images[0])

def generate_image_variation(image_path):
    print(f"Generating variation of the image: {image_path}")
    
    with open(image_path, 'rb') as image_file:
        response = openAI_client.images.create_variation(
            image=image_file,
            n=1,
            size="1024x1024"
        )

    image_url = response.data[0].url
    print(f"Variation Image URL: {image_url}")

    img_data = requests.get(image_url).content

    folder_path = create_img_folder()
    img_filename = generate_filename("Variation_Image")
    img_file_path = os.path.join(folder_path, img_filename)

    with open(img_file_path, 'wb') as handler:
        handler.write(img_data)
    
    print(f"Variation image saved to {img_file_path}")
    return img_file_path

# Main process logic
if __name__ == "__main__":
    folder_path = create_img_folder()

    # Check if there are any images in the folder
    last_image_path = get_last_image_path(folder_path)

    if last_image_path:
        # Generate a new image based on the last image in the IMG folder
        print(f"Last image found: {last_image_path}")
        img_path = generate_image_variation(last_image_path)
    else:
        # If no images, create a new image from the prompt
        print("No existing images found. Generating a new image...")
        img_path = generate_image(complete_prompt)

    print("Process complete.")