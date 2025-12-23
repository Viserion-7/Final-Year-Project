import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not loaded"

client = genai.Client()

def extract_text_from_gemini(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text.strip()

    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:
            if not candidate:
                continue
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    return text.strip()

    raise RuntimeError("No text returned from Gemini")

def split_sandhi(text: str) -> str:
    prompt = f"""You are a Sanskrit grammar expert.

Task:
Split the ENTIRE input Sanskrit text into ALL Sandhi-separated components.

Important:
- The input may contain MULTIPLE words joined by Sandhi.
- You must split the FULL INPUT, not just the first Sandhi.
- Continue splitting until the END of the input text.
- Include ALL resulting components in order.

Rules:
- Use '+' as the separator between components.
- Output EXACTLY ONE LINE.
- Output ONLY the split text.
- Use Devanagari script only.
- Do NOT explain anything.
- Do NOT stop early.

Input:
{text}
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
        ),
    )

    result = extract_text_from_gemini(response)

    if "+" not in result:
        raise ValueError(f"Incomplete Sandhi split: {result}")

    return result


if __name__ == "__main__":
    word = input("Enter Sanskrit text: ").strip()
    print(split_sandhi(word))
