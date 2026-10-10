"""
Core image processing logic for Worksheet Cleaner.
Supports:
1. Gemini AI reconstruction: Reconstructs clean, watermark-free, blank worksheets from photos.
2. Local OpenCV filtering: HSV color filtering and inpainting for high-contrast scans.
"""

import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import List, Tuple, Optional
from xhtml2pdf import pisa


class WorksheetCleaner:
    def __init__(self):
        # Default HSV bounds for blue pen ink
        self.default_blue_lower = np.array([90, 50, 50])
        self.default_blue_upper = np.array([130, 255, 255])

    # ---------------------------------------------------------
    # GEMINI AI PROCESSING
    # ---------------------------------------------------------
    def clean_with_gemini(self, img_bytes: bytes, api_key: str) -> Tuple[Optional[str], str]:
        """
        Uses Gemini Vision API to transcribe the worksheet while completely
        removing student handwriting, watermarks, and photo lighting flaws.
        Returns HTML string of the cleaned page and a status message.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None, "Error: google-genai library is not installed."

        prompt = (
            "You are an expert school worksheet transcriber and educator.\n"
            "Analyze this photo of a school assignment/workbook page carefully.\n\n"
            "TASKS:\n"
            "1. Transcribe the COMPLETE printed worksheet exactly as formatted (headers, titles, instructions, "
            "grammar tables, examples, numbers, questions, and punctuation).\n"
            "2. CRITICAL: Completely REMOVE all student handwritten answers, pen strokes, doodles, and markings.\n"
            "3. Replace every removed answer or fill-in section with a clean blank space for a new student:\n"
            "   - In tables: leave the cell blank (e.g. <td>&nbsp;</td>).\n"
            "   - In fill-in-the-blank questions: insert a clean dotted line (e.g. <span class='blank'>........................................</span>).\n"
            "4. Remove any watermarks (such as 'Al Retaj' or publisher marks), page shadows, desk backgrounds, and camera skew.\n"
            "5. Generate clean, semantic HTML with clean table borders and high-contrast black typography.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the HTML inside a single `<div class='worksheet-page'>...</div>` element.\n"
            "Do NOT wrap in markdown code blocks like ```html. Output raw HTML only."
        )

        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_error = ""

        try:
            client = genai.Client(api_key=api_key.strip())
            
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
                        html_text = response.text.strip()
                        # Clean any accidental markdown code fences
                        html_text = re.sub(r"^```(?:html)?\s*", "", html_text, flags=re.MULTILINE)
                        html_text = re.sub(r"\s*```$", "", html_text, flags=re.MULTILINE).strip()
                        return html_text, f"Success (using {model_name})"
                except Exception as e:
                    last_error = str(e)
                    continue

            return None, f"Gemini API error: {last_error}"

        except Exception as e:
            return None, f"Failed to initialize Gemini Client: {str(e)}"

    def html_to_pdf(self, pages_html: List[str]) -> bytes:
        """
        Combines a list of page HTML strings into a clean printable PDF.
        """
        css = """
        <style>
            @page {
                size: a4 portrait;
                margin: 1.5cm;
            }
            body {
                font-family: Helvetica, Arial, sans-serif;
                color: #111111;
                font-size: 13pt;
                line-height: 1.6;
            }
            .worksheet-page {
                page-break-after: always;
            }
            .worksheet-page:last-child {
                page-break-after: auto;
            }
            h1, h2, h3 {
                color: #000000;
                margin-top: 0;
                margin-bottom: 12px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            th, td {
                border: 1.5px solid #222222;
                padding: 8px 12px;
                text-align: left;
                vertical-align: middle;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .blank {
                color: #555555;
                font-weight: normal;
                letter-spacing: 2px;
            }
            ul, ol {
                margin: 10px 0;
                padding-left: 25px;
            }
            li {
                margin-bottom: 8px;
            }
        </style>
        """

        body_content = "\n".join(pages_html)
        full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{body_content}</body></html>"

        pdf_stream = io.BytesIO()
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_stream, encoding='utf-8')
        
        if pisa_status.err:
            # Fallback basic PDF
            return pdf_stream.getvalue()
        return pdf_stream.getvalue()

    # ---------------------------------------------------------
    # OPENCV LOCAL PROCESSING (FALLBACK / OFFLINE)
    # ---------------------------------------------------------
    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Enhances image quality by adjusting contrast and sharpening slightly.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

    def resize_if_large(self, img: np.ndarray, max_dim: int = 4000) -> np.ndarray:
        """
        Resizes the image if any dimension exceeds max_dim to prevent memory issues.
        """
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def remove_blue_pen(self, img: np.ndarray, sensitivity: float = 1.0) -> Tuple[np.ndarray, bool]:
        """
        Removes blue pen ink using HSV color filtering and inpainting.
        """
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
        """
        Removes any colored or light handwriting while preserving dark printed text.
        """
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
        """
        Local OpenCV fallback image processing.
        """
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
        Converts a list of BGR numpy images to a single PDF file using Pillow.
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
