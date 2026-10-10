import base64
from google import genai
import os
import dotenv
from google.genai import types
from PIL import Image
import io

import cv2
import numpy as np

img_path = "C:\\Users\\LEGION\\Pictures\\Saved Pictures\\WhatsApp Image 2026-10-10 at 1.21.50 PM.jpeg"

with open(img_path, "rb") as f:
    img_bytes = f.read()

dotenv.load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key.strip())
all_models = client.models.list()
supported = []
for m in all_models:
    name = getattr(m, 'name', str(m)).replace("models/", "")
    if any(x in name for x in ["image", "banana", "nano"]):
        supported.append(name)
print(supported)

prompt = ["Remove all handwritten pen answers from this image and return the clean questions only."]
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=[
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        prompt,
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"]
    )
)

if response and response.candidates:
    for part in response.candidates[0].content.parts:
        if getattr(part, 'inline_data', None):
            raw_data = part.inline_data.data
            if isinstance(raw_data, str):
                raw_bytes = base64.b64decode(raw_data)
            else:
                raw_bytes = raw_data

            nparr = np.frombuffer(raw_bytes, np.uint8)
            cleaned_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cleaned_bgr is not None:
                print("Successfully regenerated using " + "gemini-3.1-flash-lite")
                cv2.imwrite("./images/worksheet_clean.jpg", cleaned_bgr)
                print("Saved cleaned image to ./images/worksheet_clean.jpg")