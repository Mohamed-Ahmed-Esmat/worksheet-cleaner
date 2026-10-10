"""
Core image processing logic for Worksheet Cleaner.
Supports:
1. AI Image Eraser: Uses Gemini Vision to detect exact coordinates of handwritten answers,
   then inpaints/erases them directly from the original image, returning the exact cleaned page.
2. Local OpenCV Eraser: Traditional color thresholding for clean scans.
"""

import cv2
import numpy as np
from PIL import Image
import io
import re
import json
from typing import List, Tuple, Optional


class WorksheetCleaner:
    def __init__(self):
        # Default HSV bounds for blue pen ink
        self.default_blue_lower = np.array([90, 50, 50])
        self.default_blue_upper = np.array([130, 255, 255])

    # ---------------------------------------------------------
    # GEMINI AI IMAGE ERASER (IMAGE IN -> CLEANED IMAGE OUT)
    # ---------------------------------------------------------
    def get_available_models(self, api_key: str) -> List[str]:
        """
        Dynamically queries Google AI Studio for models available to this API key.
        """
        if not api_key or not api_key.strip():
            return []

        try:
            from google import genai
            client = genai.Client(api_key=api_key.strip())
            all_models = client.models.list()
            
            supported = []
            for m in all_models:
                name = getattr(m, 'name', str(m)).replace("models/", "")
                if "gemini" in name and not any(x in name for x in ["embedding", "aqa", "imagen"]):
                    supported.append(name)

            def score_model(m):
                score = 0
                if "flash" in m: score += 100
                if "pro" in m: score += 90
                if "3." in m: score += 30
                if "2.5" in m: score += 20
                if "2.0" in m: score += 10
                return -score

            supported.sort(key=score_model)
            return supported
        except Exception:
            return ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.5-pro", "gemini-3.0-pro", "gemini-2.0-flash"]

    def clean_image_with_gemini_ai(
        self, img_bytes: bytes, api_key: str, chosen_model: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Uses Gemini Vision to locate student handwriting/answers on the page,
        then precisely in-paints and erases them from the ORIGINAL IMAGE.
        Returns the cleaned image (BGR numpy array) and status message.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None, "Error: google-genai library is not installed."

        # Decode original image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Error: Could not decode image."

        img = self.resize_if_large(img)
        h, w = img.shape[:2]

        prompt = (
            "You are an expert document analysis AI.\n"
            "Analyze this photo of a school workbook/worksheet page.\n"
            "Detect ALL student handwriting, handwritten answers, filled-in table cells, pen scribbles, and written solutions.\n"
            "CRITICAL: Do NOT detect printed typography, printed questions, printed headers, or table lines.\n\n"
            "Return a JSON array of 2D bounding boxes enclosing each handwritten answer, normalized from 0 to 1000:\n"
            "[\n"
            '  {"box_2d": [ymin, xmin, ymax, xmax], "label": "handwriting"}\n'
            "]\n"
            "Output ONLY the raw JSON array. No explanations or markdown."
        )

        try:
            client = genai.Client(api_key=api_key.strip())
            
            models_to_try = []
            if chosen_model and chosen_model.strip():
                models_to_try.append(chosen_model.strip())
            
            available = self.get_available_models(api_key)
            for m in available:
                if m not in models_to_try:
                    models_to_try.append(m)

            if not models_to_try:
                models_to_try = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.5-pro", "gemini-3.0-pro"]

            boxes = []
            used_model = ""
            last_error = ""

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            prompt,
                        ]
                    )
                    if response and response.text:
                        raw_text = response.text.strip()
                        # Extract JSON array
                        match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
                        if match:
                            try:
                                boxes = json.loads(match.group(0))
                            except Exception:
                                pass
                        
                        if not boxes:
                            # Fallback regex for [ymin, xmin, ymax, xmax]
                            nums = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', raw_text)
                            for n in nums:
                                boxes.append({"box_2d": [int(x) for x in n]})

                        used_model = model_name
                        break
                except Exception as e:
                    last_error = str(e)
                    continue

            if not boxes and not used_model:
                return None, f"Gemini API error: {last_error}"

            # Inpaint the detected handwriting directly on the original image
            cleaned_img = self.inpaint_boxes(img, boxes)
            msg = f"Cleaned {len(boxes)} handwritten areas using {used_model}"
            return cleaned_img, msg

        except Exception as e:
            return None, f"Failed to process with Gemini: {str(e)}"

    def inpaint_boxes(self, img: np.ndarray, boxes: List[dict]) -> np.ndarray:
        """
        Creates a stroke mask inside the AI-detected bounding boxes and erases the handwriting.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for item in boxes:
            box = item.get("box_2d", [])
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                y1 = max(0, int(ymin * h / 1000.0))
                x1 = max(0, int(xmin * w / 1000.0))
                y2 = min(h, int(ymax * h / 1000.0))
                x2 = min(w, int(xmax * w / 1000.0))

                if y2 > y1 and x2 > x1:
                    roi_gray = gray[y1:y2, x1:x2]
                    # Estimate local paper brightness
                    paper_val = np.percentile(roi_gray, 80)
                    
                    # Target ink strokes that are darker than the local paper
                    stroke_mask = roi_gray < (paper_val - 20)
                    kernel = np.ones((3, 3), np.uint8)
                    stroke_uint8 = (stroke_mask.astype(np.uint8)) * 255
                    stroke_uint8 = cv2.dilate(stroke_uint8, kernel, iterations=1)

                    mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], stroke_uint8)

        if cv2.countNonZero(mask) > 0:
            cleaned = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
            return cleaned
        return img

    # ---------------------------------------------------------
    # OPENCV LOCAL PROCESSING (FALLBACK / OFFLINE)
    # ---------------------------------------------------------
    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

    def resize_if_large(self, img: np.ndarray, max_dim: int = 3000) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def remove_blue_pen(self, img: np.ndarray, sensitivity: float = 1.0) -> Tuple[np.ndarray, bool]:
        sat_lower = max(20, int(50 - (sensitivity - 1.0) * 20))
        lower_bound = np.array([90, sat_lower, 50])
        upper_bound = self.default_blue_upper

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        if cv2.countNonZero(mask) < 50:
            return img, False

        cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return cleaned, True

    def remove_multi_color_pen(self, img: np.ndarray, sensitivity: float = 1.0) -> Tuple[np.ndarray, bool]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sat_thresh = max(15, int(35 - (sensitivity - 1.0) * 15))
        sat_mask = cv2.inRange(hsv[:, :, 1], sat_thresh, 255)

        lower_gray = max(60, int(80 - (sensitivity - 1.0) * 15))
        upper_gray = min(200, int(180 + (sensitivity - 1.0) * 15))
        intensity_mask = cv2.inRange(gray, lower_gray, upper_gray)

        low_sat_mask = cv2.inRange(hsv[:, :, 1], 0, sat_thresh)
        pencil_mask = cv2.bitwise_and(intensity_mask, low_sat_mask)

        mask = cv2.bitwise_or(sat_mask, pencil_mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        if cv2.countNonZero(mask) < 50:
            return img, False

        cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return cleaned, True

    def process_image(self, img_bytes: bytes, mode: str, sensitivity: float, enhance: bool) -> Tuple[Optional[np.ndarray], str]:
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "Error: Could not decode image."

            img = self.resize_if_large(img)
            if enhance:
                img = self.enhance_image(img)

            status_msg = "Success"
            if mode == "Blue Pen Removal":
                cleaned, detected = self.remove_blue_pen(img, sensitivity)
                if not detected:
                    status_msg = "Warning: No blue ink detected."
            else:
                cleaned, detected = self.remove_multi_color_pen(img, sensitivity)
                if not detected:
                    status_msg = "Warning: No colored handwriting detected."

            return cleaned, status_msg
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def images_to_pdf(self, images_bgr: List[np.ndarray]) -> bytes:
        """
        Converts a list of BGR numpy images into a multi-page PDF using Pillow.
        """
        pil_images = []
        for img_bgr in images_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))

        if not pil_images:
            return b""

        pdf_bytes = io.BytesIO()
        pil_images[0].save(
            pdf_bytes, "PDF", resolution=100.0, save_all=True, append_images=pil_images[1:]
        )
        return pdf_bytes.getvalue()
