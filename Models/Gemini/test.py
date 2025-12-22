import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from Models/
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

# Safety check
assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not loaded"

# Create Gemini client (API key auto-read from env)
client = genai.Client()

def split_sandhi(word: str) -> str:
    prompt = f"""You are a Sanskrit grammar expert.

Task:
Fully split the given Sanskrit text into ALL Sandhi components.

Rules:
- Perform a COMPLETE Sandhi split (no partial results).
- Include ALL resulting words in order.
- Use '+' as the separator.
- Output exactly ONE line.
- Use Devanagari script only.
- Do NOT explain anything.
- Do NOT stop early.

Input:
{word}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=64,
        ),
    )

    return response.text.strip()


if __name__ == "__main__":
    word = input("Enter Sanskrit word (Devanagari): ").strip()
    if not word:
        raise ValueError("No input provided")

    print(split_sandhi(word))


# import os
# from pathlib import Path
# from google import genai
# from dotenv import load_dotenv
# env_path = Path(__file__).resolve().parents[1] / ".env"
# load_dotenv(env_path)

# # Safety check
# assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not loaded"

# client = genai.Client()

# print("Available Gemini models:\n")

# for model in client.models.list():
#     # Show only models that support text generation
#     if "generateContent" in model.supported_actions:
#         print(model.name)
