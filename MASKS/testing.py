import openai

openai.api_key = ""

response = openai.Image.create_edit(
    image=open("drawing.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Paint with color",
    n=1,
    size="512x512",
)
image_url = response["data"][0]["url"]

print(image_url) 