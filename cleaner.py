"""
Core image processing logic for Worksheet Cleaner.
Uses OpenCV to detect and remove handwritten pen ink from worksheets.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Optional

class WorksheetCleaner:
    def __init__(self):
        # Default HSV bounds for blue pen ink
        self.default_blue_lower = np.array([90, 50, 50])
        self.default_blue_upper = np.array([130, 255, 255])
    
    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Enhances image quality by adjusting contrast and sharpening slightly.
        This helps in better detection of pen marks and cleaner final output.
        """
        # Convert to LAB color space for better contrast enhancement on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original with sharpened to avoid artifacts
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
        Sensitivity controls how wide the hue/saturation range is.
        Returns the cleaned image and a boolean indicating if blue was detected.
        """
        # Adjust saturation bound based on sensitivity (higher sensitivity = lower saturation threshold)
        sat_lower = max(20, int(50 - (sensitivity - 1.0) * 20))
        lower_bound = np.array([90, sat_lower, 50])
        upper_bound = self.default_blue_upper
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Dilate mask slightly to catch edges of the pen strokes where color fades
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Check if blue ink was detected
        if cv2.countNonZero(mask) < 50:
            return img, False
            
        # Inpaint using Telea method to fill the masked regions with surrounding pixels
        cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return cleaned, True

    def remove_multi_color_pen(self, img: np.ndarray, sensitivity: float = 1.0) -> Tuple[np.ndarray, bool]:
        """
        Fallback mode: Removes any colored or light handwriting while preserving dark printed text.
        
        Strategy:
        - Colored pen (blue, red, green, etc.): Detected via saturation in HSV space.
          Printed black text has near-zero saturation, while colored ink is highly saturated.
        - Light pencil/gray pen: Detected via a narrow intensity band.
          Printed text is very dark (< ~80), paper is very light (> ~220).
          Handwriting in pencil/light pen falls in between (~80-180).
        
        Returns the cleaned image and a boolean indicating success.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Saturation-based mask: catches any colored ink (blue, red, green, etc.)
        # Printed black text has very low saturation, so this is safe
        sat_thresh = max(15, int(35 - (sensitivity - 1.0) * 15))
        sat_mask = cv2.inRange(hsv[:,:,1], sat_thresh, 255)
        
        # 2. Intensity-based mask: catches light pencil / gray ink
        # Only target the "middle" intensity range to avoid removing printed text (dark)
        # or paper background (light)
        lower_gray = max(60, int(80 - (sensitivity - 1.0) * 15))
        upper_gray = min(200, int(180 + (sensitivity - 1.0) * 15))
        intensity_mask = cv2.inRange(gray, lower_gray, upper_gray)
        
        # Only apply intensity mask where saturation is low (to avoid double-counting)
        # This targets gray/pencil marks specifically
        low_sat_mask = cv2.inRange(hsv[:,:,1], 0, sat_thresh)
        pencil_mask = cv2.bitwise_and(intensity_mask, low_sat_mask)
        
        # Combine: colored ink OR pencil marks
        mask = cv2.bitwise_or(sat_mask, pencil_mask)
        
        # Clean up the mask: dilate to catch edges, then apply small morphological close
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if cv2.countNonZero(mask) < 50:
            return img, False
            
        cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return cleaned, True

    def process_image(self, img_bytes: bytes, mode: str, sensitivity: float, enhance: bool) -> Tuple[Optional[np.ndarray], str]:
        """
        Processes a single image based on the selected mode and parameters.
        Returns the processed image (BGR) and a status message.
        """
        try:
            # Decode image from bytes
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Error: Could not decode image. Format might be unsupported (e.g., HEIC)."
                
            img = self.resize_if_large(img)
            
            if enhance:
                img = self.enhance_image(img)
                
            status_msg = "Success"
            if mode == "Blue Pen Removal":
                cleaned, detected = self.remove_blue_pen(img, sensitivity)
                if not detected:
                    status_msg = "Warning: No blue ink detected. You might want to try 'Auto-Detect (Any Color)' mode."
            else:
                cleaned, detected = self.remove_multi_color_pen(img, sensitivity)
                if not detected:
                    status_msg = "Warning: No colored handwriting detected."
                    
            return cleaned, status_msg
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def images_to_pdf(self, images_bgr: List[np.ndarray]) -> bytes:
        """
        Converts a list of BGR numpy images to a single PDF file (returned as bytes).
        """
        pil_images = []
        for img_bgr in images_bgr:
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))
            
        if not pil_images:
            return b""
            
        pdf_bytes = io.BytesIO()
        # Save first image and append the rest
        pil_images[0].save(
            pdf_bytes, "PDF", resolution=100.0, save_all=True, append_images=pil_images[1:]
        )
        return pdf_bytes.getvalue()
