"""
Core image processing logic for Worksheet Cleaner.
Supports:
1. Native Gemini Image Regeneration (Nano Banana / Flash Image):
   Uses Gemini's native image-to-image multimodal generation to REGENERATE
   the exact worksheet image cleanly without the handwritten answers.
2. AI Bounding Box Inpainting (Fallback).
3. Local OpenCV Filter (Offline).
"""

import cv2
import numpy as np
from PIL import Image
import io
import re
import json
import base64
from typing import List, Tuple, Optional


class WorksheetCleaner:
    def __init__(self):
        # Default HSV bounds for blue pen ink
        self.default_blue_lower = np.array([90, 50, 50])
        self.default_blue_upper = np.array([130, 255, 255])

    # ---------------------------------------------------------
    # 1. NATIVE GEMINI IMAGE REGENERATION (NANO BANANA / FLASH-IMAGE)
    # ---------------------------------------------------------
    def get_available_image_models(self, api_key: str) -> List[str]:
        """
        Dynamically finds available Gemini Image models (Nano Banana / Flash Image family).
        """
        preferred = [
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image",
        ]

        if not api_key or not api_key.strip():
            return preferred

        try:
            from google import genai
            client = genai.Client(api_key=api_key.strip())
            all_models = client.models.list()
            supported = []
            for m in all_models:
                name = getattr(m, 'name', str(m)).replace("models/", "")
                if any(x in name for x in ["image", "banana", "nano"]):
                    supported.append(name)

            if supported:
                return supported
        except Exception:
            pass

        return preferred

    def regenerate_image_with_ai(
        self,
        img_bytes: bytes,
        api_key: str,
        chosen_model: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Calls Gemini Image Generation (Nano Banana / Flash-Image) to take the
        worksheet photo and directly REGENERATE the clean page image with all
        answers cleanly removed.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None, "Error: google-genai library is not installed."

        client = genai.Client(api_key=api_key.strip())

        default_prompt = (
            "Clean and regenerate this school worksheet page as a clean, high-contrast digital image.\n"
            "Preserve 100% of all printed content: keep all printed text, headings, exercise instructions, "
            "vocabulary, grammar tables, question numbers, and dotted lines completely intact and unchanged.\n"
            "CRITICAL: Erase and remove ONLY the handwritten pen/pencil answers, markings, and student doodles.\n"
            "Ensure the answer spaces (dotted lines, blank lines, empty table cells) are clean, blank, and empty so a student can re-practice.\n"
            "Remove any background desk shadows, page skew, or diagonal watermarks.\n"
            "Output the result as a clear, crisp printable worksheet image."
        )

        prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else default_prompt

        models_to_try = []
        if chosen_model and chosen_model.strip():
            models_to_try.append(chosen_model.strip())

        defaults = [
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image",
        ]
        for m in defaults:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = ""

        # Attempt 1: Direct native image regeneration
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
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
                                return cleaned_bgr, f"Successfully regenerated using {model_name}"

            except Exception as e:
                last_error = f"{model_name}: {str(e)}"
                continue

        # Attempt 2: If image generation modality isn't supported on current key,
        # fallback to high-precision coordinate inpainting
        fallback_img, fallback_msg = self.fallback_coordinate_inpaint(img_bytes, client)
        if fallback_img is not None:
            return fallback_img, f"Regenerated via AI Inpainting (image model fallback: {last_error})"

        return None, f"Image regeneration failed: {last_error}"

    def fallback_coordinate_inpaint(self, img_bytes: bytes, client) -> Tuple[Optional[np.ndarray], str]:
        """
        Fallback inpainting using multimodal coordinate detection.
        """
        try:
            from google.genai import types
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "Decode error"

            img = self.resize_if_large(img)
            h, w = img.shape[:2]

            prompt = (
                "Detect all handwritten text, student answers, and pen markings in this image. "
                "Do NOT detect printed text or table grids. "
                "Return a JSON array of bounding boxes normalized from 0 to 1000: "
                '[{"box_2d": [ymin, xmin, ymax, xmax]}]'
            )

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt,
                ]
            )

            raw_text = response.text if response else ""
            boxes = []
            nums = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', raw_text)
            for n in nums:
                boxes.append({"box_2d": [int(x) for x in n]})

            mask = np.zeros((h, w), dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for b in boxes:
                box = b.get("box_2d", [])
                if len(box) == 4:
                    y1 = max(0, int(box[0] * h / 1000.0))
                    x1 = max(0, int(box[1] * w / 1000.0))
                    y2 = min(h, int(box[2] * h / 1000.0))
                    x2 = min(w, int(box[3] * w / 1000.0))
                    if y2 > y1 and x2 > x1:
                        roi = gray[y1:y2, x1:x2]
                        pval = np.percentile(roi, 80)
                        smask = (roi < (pval - 20)).astype(np.uint8) * 255
                        kernel = np.ones((3, 3), np.uint8)
                        smask = cv2.dilate(smask, kernel, iterations=1)
                        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], smask)

            if cv2.countNonZero(mask) > 0:
                cleaned = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
                return cleaned, "Cleaned via coordinate inpainting"
            return img, "No handwriting detected"
        except Exception as e:
            return None, str(e)

    # ---------------------------------------------------------
    # 2. LOCAL OPENCV PROCESSING (OFFLINE)
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
