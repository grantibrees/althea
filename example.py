import openai
import os

# .env
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("api_key")
openai.api_key = key

from fileopener import open_file, save_file


blogpost = open_file("textexamples/blogpost.txt")
prompt1 = open_file("textexamples/prompt1.txt").replace("<<FEED>>", blogpost)

input = open_file("textexamples/input.txt")
summary = open_file("textexamples/summary.txt").replace("<<INPUT>>", input)

inputai = prompt1


response = openai.Completion.create(
    model="text-davinci-003",
    prompt=inputai,
    temperature=1.0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

text = response["choices"][0]["text"].strip()
print(text)

# Save output:
save_file("textexamples/blogpostideas.txt", text)