from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction="Atue como um especialista em machine learning."
    ),
    contents="Explique de forma resumida como a IA funciona",
)
print(response.text)
