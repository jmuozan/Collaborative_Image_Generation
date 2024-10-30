from openai import OpenAI
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


response = client.images.edit(
    image=open("k.png", "rb"),
    mask=open("k_mask.png", "rb"),
    prompt="fill with elephants",
    n=1,
    size="512x512",
)
image_url = response.data[0].url

print(image_url)