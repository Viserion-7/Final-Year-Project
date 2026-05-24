from PIL import Image
import os

os.makedirs("dpi600", exist_ok=True)

for file in os.listdir("."):
    if file.lower().endswith((".jpg")):
        img = Image.open(file)
        img.save(f"dpi600/{file}", dpi=(600,600))

print("All images saved in dpi600 folder.")