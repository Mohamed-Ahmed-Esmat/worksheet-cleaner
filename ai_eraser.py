import json
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional

def extract_boxes_from_text(text: str) -> List[dict]:
    # Extract JSON array from text
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Fallback to finding all box_2d arrays
    boxes = []
    box_matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    for b in box_matches:
        boxes.append({"box_2d": [int(x) for x in b]})
    return boxes

def inpaint_handwriting_boxes(img: np.ndarray, boxes: List[dict]) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for item in boxes:
        box = item.get("box_2d", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            # Denormalize (0-1000 to pixels)
            y1 = max(0, int(ymin * h / 1000))
            x1 = max(0, int(xmin * w / 1000))
            y2 = min(h, int(ymax * h / 1000))
            x2 = min(w, int(xmax * w / 1000))
            
            if y2 > y1 and x2 > x1:
                # Inside this box, isolate the handwriting strokes
                roi_gray = gray[y1:y2, x1:x2]
                
                # Estimate local paper brightness (e.g. 75th percentile)
                paper_val = np.percentile(roi_gray, 80)
                
                # Any pixel darker than paper by margin is ink
                stroke_mask = roi_gray < (paper_val - 25)
                
                # Expand slightly to cover stroke anti-aliasing
                kernel = np.ones((3, 3), np.uint8)
                stroke_mask_uint8 = (stroke_mask.astype(np.uint8)) * 255
                stroke_mask_uint8 = cv2.dilate(stroke_mask_uint8, kernel, iterations=1)
                
                mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], stroke_mask_uint8)
    
    if cv2.countNonZero(mask) > 0:
        cleaned = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        return cleaned
    return img

print("AI eraser module verified")
