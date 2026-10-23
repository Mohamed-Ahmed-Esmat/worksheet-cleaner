"""
Detect-then-erase engine.

A cheap text-output vision model (Gemini Flash-Lite, called on the Gemini API
directly or via OpenRouter) only *locates* handwriting as bounding boxes. The erasing is done locally with
OpenCV, so printed text is never redrawn and cannot be altered.

Pen vs. print is decided per pixel by color, measured as log channel ratios
relative to the paper (so it does not depend on brightness: very dark navy
ink is still clearly "blue", whereas Lab chroma collapses towards zero for
dark pixels). What printed ink looks like is learned from the ink just
around each box, so any pen color and any camera color cast work.

The model also labels each box:
- "answer": handwriting in empty space. All ink in the box is erased (so
  pencil and black pen work too), except long printed lines such as table
  borders; pen-colored pixels are also caught in a small margin.
- "mark": a circle/underline drawn over printed text. Only pen-colored
  pixels are erased, so the printed word underneath is kept (where the pen
  crossed a letter, that letter gets a small gap).
"""

import base64
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

from gemini_client import GeminiError, call_gemini, image_part

PROVIDER_GEMINI = "gemini"          # Google Gemini API directly (has a free tier)
PROVIDER_OPENROUTER = "openrouter"  # same models via OpenRouter (paid, no geo-blocks)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DETECTOR_MODEL = "google/gemini-3.5-flash-lite"

DETECTION_PROMPT = """\
This is a photo of a printed school worksheet that a student has filled in by hand.
Find every handwritten mark the student added: written answers, filled-in words or \
letters, table entries, circles, underlines, ticks, crosses, arrows and scribbles.
Do NOT include printed text, printed lines, dotted answer lines, table borders, \
page numbers or watermarks.

Return JSON: {"marks": [{"box_2d": [ymin, xmin, ymax, xmax], "type": "answer" | "mark"}]}
- box_2d is normalized to 0-1000 and must tightly enclose the handwriting.
- "answer": handwriting written in empty space (it does not overlap printed text).
- "mark": a circle, underline, tick or cross drawn over or around printed text.
Return {"marks": []} if there is no handwriting."""

# Tuning constants
_INK_DARKNESS = 0.82     # pixel is ink if darker than this fraction of the local paper
_BOX_PAD = 0.006         # boxes are approximate: pad by this fraction of the page size
_LINE_LENGTH = 0.04      # straight runs longer than this fraction of the width are printed lines
_ANSWER_TRIM = 0.06      # sides of an "answer" box where only pen-colored pixels are erased
_PEN_DIRECTION_DISTANCE = 8.0  # off-color pixels used to learn the pen's color direction
_PEN_DISTANCE = 5.0      # color distance (in print standard deviations) that marks a pen pixel
_PEN_GROW = 2.5          # pen strokes extend through neighbours at least this far from print
_MIN_SEED_AREA = 4       # confident pen blobs smaller than this many pixels are noise
_STROKE_REACH = 0.5      # a stroke may be followed this far (fraction of box size) past its box
_MIN_COLOR_SPREAD = 0.01 # floor for the print color spread, so clean scans do not over-trigger
_DARK_OFFSET = 12.0      # added to 0-255 channel values before taking color ratios
_LUM_BINS = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.82, 0.92, 1.0, np.inf])


@dataclass
class Mark:
    """A detected handwriting region in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: str  # "answer" or "mark"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_marks(
    jpeg_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_DETECTOR_MODEL,
    provider: str = PROVIDER_OPENROUTER,
) -> str:
    """
    Asks the vision model for handwriting boxes. Returns the raw text reply.
    Model ids may be given with or without OpenRouter's "google/" prefix.
    """
    if provider == PROVIDER_GEMINI:
        return _detect_with_gemini(jpeg_bytes, api_key, model.removeprefix("google/"))
    if "/" not in model:
        model = "google/" + model
    return _detect_with_openrouter(jpeg_bytes, api_key, model)


def _detect_with_gemini(jpeg_bytes: bytes, api_key: str, model: str) -> str:
    try:
        return call_gemini(
            [image_part(jpeg_bytes), {"text": DETECTION_PROMPT}],
            api_key,
            model,
            {"temperature": 0, "response_mime_type": "application/json"},
            timeout=120,
        )
    except GeminiError as e:
        if e.is_quota:
            raise GeminiError(f"{e} Wait and retry, or switch the provider to OpenRouter.", 429) from e
        raise


def _detect_with_openrouter(jpeg_bytes: bytes, api_key: str, model: str) -> str:
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "HTTP-Referer": "https://worksheet-cleaner.streamlit.app",
        "X-Title": "Worksheet Cleaner",
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {}).get("message", resp.text)
        except ValueError:
            err = resp.text
        raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {err}")

    content = resp.json()["choices"][0]["message"].get("content") or ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


_BOX_RE = re.compile(
    r'"box_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    r'(?:[^{}\[\]]*?"type"\s*:\s*"(\w+)")?'
)


def parse_marks(text: str, width: int, height: int) -> List[Mark]:
    """
    Converts the model's 0-1000 [ymin, xmin, ymax, xmax] boxes to pixel Marks.
    Uses a regex instead of json.loads because Flash-Lite models sometimes
    emit slightly malformed JSON (e.g. a stray closing brace).
    """
    marks = []
    for ymin, xmin, ymax, xmax, kind in _BOX_RE.findall(text):
        y1, x1, y2, x2 = (int(v) for v in (ymin, xmin, ymax, xmax))
        x1, x2 = sorted((x1 * width // 1000, x2 * width // 1000))
        y1, y2 = sorted((y1 * height // 1000, y2 * height // 1000))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            marks.append(Mark(x1, y1, x2, y2, "answer" if kind == "answer" else "mark"))
    return marks


# ---------------------------------------------------------------------------
# Local erasing
# ---------------------------------------------------------------------------
def estimate_paper(img: np.ndarray) -> np.ndarray:
    """Paper color at every pixel (text removed by a max filter, then smoothed)."""
    k = max(7, (min(img.shape[:2]) // 80) | 1)
    paper = cv2.dilate(img, np.ones((k, k), np.uint8))
    return cv2.medianBlur(paper, k * 2 + 1)


def flatten_lighting(img: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Divides out uneven lighting so the paper becomes white (a 'scanned' look)."""
    return cv2.divide(img, paper, scale=255)


def _printed_lines(ink: np.ndarray) -> np.ndarray:
    """Long horizontal/vertical runs of ink: table borders and printed rules."""
    n = max(15, int(ink.shape[1] * _LINE_LENGTH))
    ink_u8 = ink.astype(np.uint8) * 255
    horiz = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, np.ones((1, n), np.uint8))
    vert = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, np.ones((n, 1), np.uint8))
    return (horiz | vert) > 0


def _log_chromaticity(img: np.ndarray, paper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel (log B/R, log G/R) relative to the paper, plus relative luminance."""
    # The offset keeps the ratios stable for near-black pixels, where a
    # channel value of 1 vs 3 would otherwise look like a huge color shift.
    rel = (img.astype(np.float32) + _DARK_OFFSET) / (paper.astype(np.float32) + _DARK_OFFSET)
    b, g, r = rel[..., 0], rel[..., 1], rel[..., 2]
    chroma = np.stack([np.log(b / r), np.log(g / r)], axis=-1)
    return chroma, (b + g + r) / 3


def color_deviation(
    chroma: np.ndarray, lum: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """
    Each pixel's color offset from printed ink/paper of the same brightness, in
    standard deviations (one value per color axis). Statistics come from the
    `reference` pixels (known to be print or paper); they are binned by
    brightness because dark print picks up a slight color cast of its own.
    """
    dev = np.zeros(chroma.shape, dtype=np.float32)
    bins = np.digitize(lum, _LUM_BINS) - 1
    for b in range(len(_LUM_BINS) - 1):
        in_bin = bins == b
        ref = chroma[in_bin & reference]
        if len(ref) < 30:
            ref = chroma[reference]  # too few samples: fall back to all reference pixels
        mean = np.median(ref, axis=0)
        # Median absolute deviation: robust to stray pen or background pixels
        # that end up in the reference area.
        spread = np.maximum(1.4826 * np.median(np.abs(ref - mean), axis=0), _MIN_COLOR_SPREAD)
        dev[in_bin] = (chroma[in_bin] - mean) / spread
    return dev


def pen_score(dev: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Projects the color deviation onto the pen's own color direction, learned
    from clearly off-color `candidates` pixels. Thin, faint strokes separate
    from print much better along that one direction than by total distance.
    Falls back to the total distance when there are too few candidates.
    """
    dist = np.linalg.norm(dev, axis=-1)
    strong = candidates & (dist > _PEN_DIRECTION_DISTANCE)
    if strong.sum() < 20:
        return dist
    direction = np.median(dev[strong], axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return dist
    return dev @ (direction / norm)


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Removes connected blobs smaller than min_area (isolated JPEG noise).
    Unlike a morphological opening, this keeps 1-pixel-wide pen strokes."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)
    return np.isin(labels, big[big != 0])


def build_erase_mask(img: np.ndarray, paper: np.ndarray, marks: List[Mark]) -> np.ndarray:
    """Pixels to repaint with paper, restricted to the detected boxes."""
    h, w = img.shape[:2]
    chroma, lum = _log_chromaticity(img, paper)
    ink = lum < _INK_DARKNESS
    lines = _printed_lines(ink)

    pad = int(max(h, w) * _BOX_PAD)
    boxes = [
        (max(0, m.y1 - pad), min(h, m.y2 + pad), max(0, m.x1 - pad), min(w, m.x2 + pad))
        for m in marks
    ]
    # Reference = the page around the boxes but outside them: print and paper
    # under the same lighting, with no handwriting.
    near = np.zeros((h, w), dtype=bool)
    inside = np.zeros((h, w), dtype=bool)
    for y1, y2, x1, x2 in boxes:
        my, mx = 3 * (y2 - y1), 3 * (x2 - x1)
        near[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx] = True
        inside[y1:y2, x1:x2] = True
    dist = pen_score(color_deviation(chroma, lum, near & ~inside), inside)

    # Pen strokes = connected off-color regions (dist > _PEN_GROW, which also
    # takes in soft stroke edges) that contain confident pen pixels inside a
    # detected box. A stroke is followed past the box edge, up to a margin,
    # because the model's boxes are approximate and often clip circles.
    seeds = _drop_specks(dist > _PEN_DISTANCE, min_area=_MIN_SEED_AREA)
    weak = cv2.dilate((dist > _PEN_GROW).astype(np.uint8), np.ones((2, 2), np.uint8))
    _, labels = cv2.connectedComponents(weak, connectivity=8)
    reach = np.zeros((h, w), dtype=bool)
    for y1, y2, x1, x2 in boxes:
        my = int((y2 - y1) * _STROKE_REACH)
        mx = int((x2 - x1) * _STROKE_REACH)
        reach[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx] = True
    keep = np.unique(labels[seeds & inside])
    keep = keep[keep != 0]
    pen = np.isin(labels, keep) & reach & (dist > _PEN_GROW)
    # Grow one pixel into the soft, half-colored stroke edges.
    pen = (cv2.dilate(pen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (
        dist > _PEN_GROW * 0.75
    )

    mask = pen.copy()
    for m, (y1, y2, x1, x2) in zip(marks, boxes):
        if m.kind == "answer":
            # Written in empty space: erase every stroke whatever its color,
            # except printed lines. The box sides are trimmed because the box
            # edge tends to clip the neighbouring printed word.
            tx = int((m.x2 - m.x1) * _ANSWER_TRIM)
            core = (slice(m.y1, m.y2), slice(m.x1 + tx, m.x2 - tx))
            mask[core] |= ink[core] & ~lines[core]
    return mask


def erase_marks(
    img: np.ndarray, marks: List[Mark], scan_look: bool = True
) -> np.ndarray:
    """Repaints handwriting pixels with the paper color."""
    paper = estimate_paper(img)
    mask = build_erase_mask(img, paper, marks)
    if scan_look:
        out = flatten_lighting(img, paper)
        out[mask] = 255
        return out
    out = img.copy()
    out[mask] = paper[mask]
    return out


def draw_marks(img: np.ndarray, marks: List[Mark]) -> np.ndarray:
    """Debug overlay: green = answer, orange = mark."""
    out = img.copy()
    thickness = max(1, max(img.shape[:2]) // 600)
    for m in marks:
        color = (0, 170, 0) if m.kind == "answer" else (0, 140, 255)
        cv2.rectangle(out, (m.x1, m.y1), (m.x2, m.y2), color, thickness)
    return out


def detect_and_erase(
    jpeg_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_DETECTOR_MODEL,
    scan_look: bool = True,
    provider: str = PROVIDER_OPENROUTER,
) -> Tuple[Optional[np.ndarray], List[Mark], str]:
    """Full pipeline. Returns (cleaned BGR image or None, marks, status message)."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, [], "Error: Could not decode image."
    try:
        reply = detect_marks(jpeg_bytes, api_key, model, provider)
    except (RuntimeError, requests.RequestException, KeyError, IndexError, ValueError) as e:
        return None, [], f"Detection failed: {e}"

    marks = parse_marks(reply, img.shape[1], img.shape[0])
    if not marks and '"marks"' not in reply:
        return None, [], f"Model '{model}' did not return boxes: {reply.strip()[:250]}"

    cleaned = erase_marks(img, marks, scan_look=scan_look)
    if not marks:
        return cleaned, [], "Warning: No handwriting detected."
    return cleaned, marks, f"Erased {len(marks)} handwriting region(s) using {model}"
