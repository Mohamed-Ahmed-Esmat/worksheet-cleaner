import os
import re
import cv2
import numpy as np
import dotenv
from google import genai
from google.genai import types
from cleaner import WorksheetCleaner

dotenv.load_dotenv(r"D:\Playground\worksheet-cleaner\.env")
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key.strip())

img_path = r"C:\Users\LEGION\Pictures\Saved Pictures\WhatsApp Image 2026-10-10 at 1.21.50 PM.jpeg"
with open(img_path, "rb") as f:
    img_bytes = f.read()

prompt = (
    "Detect all student handwriting, filled-in table answers, pen markings, and written responses on this worksheet. "
    "Do not detect printed question text or table borders. "
    "Return a JSON array of 2D bounding boxes normalized from 0 to 1000: "
    '[{"box_2d": [ymin, xmin, ymax, xmax]}]'
)

print("Sending request to gemini-3.1-flash-lite (Free Tier)...")
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=[
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        prompt
    ]
)

print("Response text:\n", response.text)

boxes = []
nums = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', response.text)
for n in nums:
    boxes.append({'box_2d': [int(x) for x in n]})

print(f"Parsed {len(boxes)} handwriting boxes.")

c = WorksheetCleaner()
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cleaned = c.inpaint_boxes(img, boxes)
out_path = r"D:\Playground\worksheet-cleaner\test_ai_cleaned.jpg"
cv2.imwrite(out_path, cleaned)
print(f"SUCCESS! Saved cleaned image to {out_path}")
