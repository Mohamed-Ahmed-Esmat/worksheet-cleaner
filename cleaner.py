"""
Core image processing logic for Worksheet Cleaner.
Supports:
1. OpenRouter API: Calls Google Nano Banana Pro (google/gemini-3-pro-image)
   and Nano Banana 2.1 via OpenRouter's global API to directly regenerate
   clean worksheet images without regional/country restrictions.
2. Local OpenCV filter (Offline).
"""

import cv2
import numpy as np
from PIL import Image
import io
import re
import base64
import requests
from typing import List, Tuple, Optional


class WorksheetCleaner:
    def __init__(self):
        # Default HSV bounds for blue pen ink (offline fallback)
        self.default_blue_lower = np.array([90, 50, 50])
        self.default_blue_upper = np.array([130, 255, 255])

    # ---------------------------------------------------------
    # 1. OPENROUTER API (NANO BANANA PRO & GEMINI IMAGE MODELS)
    # ---------------------------------------------------------
    def regenerate_image_with_openrouter(
        self,
        img_bytes: bytes,
        api_key: str,
        chosen_model: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Sends the worksheet photo to OpenRouter to run Nano Banana Pro
        (google/gemini-3-pro-image) and return the regenerated clean image.
        Uses pure HTTP requests (curl style).
        """
        if not api_key or not api_key.strip():
            return None, "Error: OpenRouter API key is missing. Please enter your key in the sidebar."

        # Model selection
        model = chosen_model.strip() if chosen_model and chosen_model.strip() else "google/gemini-3-pro-image"

        # Default prompt
        default_prompt = "Remove all handwritten pen answers from this image and return the clean questions only."
        prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else default_prompt

        # Encode image to base64
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_image}"

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://worksheet-cleaner.streamlit.app",
            "X-Title": "Worksheet Cleaner",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "modalities": ["image", "text"],
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text
                return None, f"OpenRouter API error ({resp.status_code}): {err_msg}"

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None, f"No response choices returned from OpenRouter: {data}"

            # Extract generated image bytes from OpenRouter response
            cleaned_bytes = self.extract_image_bytes_from_response(data)

            if not cleaned_bytes:
                # Log snippet for debugging if image is missing
                msg_obj = choices[0].get("message", {})
                snippet = str(msg_obj)[:300]
                return None, f"Could not find image in model response: {snippet}"

            # Convert image bytes to BGR numpy array
            nparr = np.frombuffer(cleaned_bytes, np.uint8)
            cleaned_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if cleaned_bgr is None:
                return None, "Error: Could not decode regenerated image bytes."

            return cleaned_bgr, f"Successfully regenerated using {model}"

        except requests.exceptions.Timeout:
            return None, "Error: OpenRouter request timed out after 120 seconds. Please try again."
        except Exception as e:
            return None, f"Request error: {str(e)}"

    def extract_image_bytes_from_response(self, data: dict) -> Optional[bytes]:
        """
        Extracts image bytes from all possible OpenRouter response structures:
        1. OpenRouter Image API: data.data[0].b64_json or url
        2. Chat completions: message.images[0].image_url.url
        3. Chat completions: message.content (string or parts list)
        """
        # 1. OpenRouter /api/v1/images format
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            first = data["data"][0]
            if isinstance(first, dict):
                if "b64_json" in first and first["b64_json"]:
                    return base64.b64decode(first["b64_json"])
                if "url" in first and first["url"]:
                    return self._fetch_or_decode_url(first["url"])

        # 2. Chat completions format
        choices = data.get("choices", [])
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})

            # Priority A: msg.images (OpenRouter image generation standard)
            images = msg.get("images", [])
            if images and isinstance(images, list):
                for img_item in images:
                    res = self._parse_image_item(img_item)
                    if res:
                        return res

            # Priority B: choices[0].images
            choice_images = choices[0].get("images", [])
            if choice_images and isinstance(choice_images, list):
                for img_item in choice_images:
                    res = self._parse_image_item(img_item)
                    if res:
                        return res

            # Priority C: msg.content
            content = msg.get("content")
            if content:
                res = self._parse_image_item(content)
                if res:
                    return res

        return None

    def _parse_image_item(self, item) -> Optional[bytes]:
        """
        Recursively extracts image bytes from an item (dict, list, or string).
        """
        if isinstance(item, dict):
            if "image_url" in item:
                val = item["image_url"]
                if isinstance(val, dict):
                    return self._fetch_or_decode_url(val.get("url", ""))
                elif isinstance(val, str):
                    return self._fetch_or_decode_url(val)
            if "url" in item:
                return self._fetch_or_decode_url(item["url"])
            if "b64_json" in item and item["b64_json"]:
                try:
                    return base64.b64decode(item["b64_json"])
                except Exception:
                    pass
            if "image" in item:
                return self._parse_image_item(item["image"])

        elif isinstance(item, list):
            for sub in item:
                res = self._parse_image_item(sub)
                if res:
                    return res

        elif isinstance(item, str):
            return self._fetch_or_decode_url(item)

        return None

    def _fetch_or_decode_url(self, s: str) -> Optional[bytes]:
        """
        Decodes base64 data URL or downloads remote HTTP image URL.
        """
        if not s or not isinstance(s, str):
            return None
        s = s.strip()

        # Data URL
        m = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", s)
        if m:
            try:
                return base64.b64decode(m.group(1))
            except Exception:
                pass

        # Markdown image ![...](https://...)
        m_md = re.search(r"!\[.*?\]\((https?://[^\s\)]+)\)", s)
        if m_md:
            try:
                resp = requests.get(m_md.group(1), timeout=30)
                if resp.status_code == 200:
                    return resp.content
            except Exception:
                pass

        # Direct HTTP/HTTPS URL
        if s.startswith("http://") or s.startswith("https://"):
            try:
                resp = requests.get(s, timeout=30)
                if resp.status_code == 200:
                    return resp.content
            except Exception:
                pass

        # Raw base64 string
        if len(s) > 1000 and not s.startswith("<") and not s.startswith("{"):
            try:
                return base64.b64decode(s)
            except Exception:
                pass

        return None

    # ---------------------------------------------------------
    # 2. LOCAL OPENCV PROCESSING (OFFLINE FALLBACK)
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
