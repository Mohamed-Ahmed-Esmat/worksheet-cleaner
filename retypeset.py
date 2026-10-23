"""
Re-typeset engine.

Instead of erasing ink from the photo, a free vision model reads only the
PRINTED text of the worksheet and returns it as simple HTML, which is then
rendered as a crisp A4 PDF. Handwriting, pen circles, shadows and watermarks
simply never make it into the output, and no printed letter is redrawn as
pixels.

Pictures, drawings and diagrams are cropped from the photo (the model only
gives their position), cleaned to a white background and embedded in the page.
The model never controls a URL: every image is a data: URI built here.

The risk is the opposite one: a reader can silently misread a printed word
(typically one hidden under a student's circle). So each page is, by default,
read a second time by a different model and every line where the two readers
disagree is reported for the user to check.

Models are tried in order (RETYPE_CHAIN); a model that is out of quota,
overloaded or retired is skipped and the next one is used.
"""

import base64
import html as html_lib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from gemini_client import GeminiError, call_gemini, image_part

# Best first. Robotics-ER 2 was the most faithful reader in testing (3-6 s per
# page, free tier ~20 pages/day); Gemini 3 Flash is a fast free fallback;
# Gemma 4 is free with no paid tier but slower (10-27 s) and often busy.
RETYPE_CHAIN = (
    "gemini-robotics-er-2-preview",
    "gemini-3-flash-preview",
    "gemma-4-26b-a4b-it",
)

RETYPE_PROMPT = """\
This photo shows a printed school worksheet that a student has filled in by hand.
Re-create the BLANK worksheet as HTML, so it can be printed again.

Rules:
1. Transcribe ONLY the PRINTED text, exactly as printed, character by character: keep \
umlauts (ä ö ü Ä Ö Ü), ß, capitalization, punctuation, numbering (e.g. "1-Das ist") and \
the dotted answer blanks ("........"). Never fix, translate or complete printed text.
2. Remove ALL handwriting. Handwritten answers written in a blank are dropped: keep only \
the printed dots/line of that blank. Handwritten words in table cells are dropped: keep \
the empty cell. Circles, underlines, ticks or crosses drawn around/over a printed word \
are dropped, but the printed word itself is kept as normal text.
3. Ignore the big grey diagonal watermark, the photo background, the page border, and any \
text cut off at the edge of the photo or belonging to the facing page.
4. Keep the layout: use <h1>/<h2>/<h3> for headings, <b> and <u> where the print is \
bold/underlined, one <p> per numbered line, and <table> for tables. Keep every row and \
every column of a table, including empty rows and empty columns. Use <p class="num"> for \
the page number. Use only these tags: h1 h2 h3 p b u i br div table tr td th ul ol li.
5. If the worksheet contains a picture, drawing or diagram, output \
<img data-box="ymin,xmin,ymax,xmax" data-handwriting="yes|no"> at its place. The box \
tightly encloses only the picture (not the text around it) and is normalized to 0-1000 \
on this photo. data-handwriting is "yes" if the student wrote, drew or marked anything \
on or inside the picture.

First list every piece of handwriting you see (pen/pencil answers, words written on \
dotted lines or in table cells, circles, crossings-out), quoting the handwritten words and \
the printed text right next to them. Then write the HTML of the blank worksheet, in which \
none of those handwritten words appear.
Return JSON: {"handwriting": [{"near_printed_text": "...", "handwritten": "..."}], \
"html": "<body content only, no <html>, <head> or CSS>"}"""

PAGE_CSS = """
@page { size: a4 portrait; margin: 1.8cm 2cm; }
body { font-family: Times-Roman; font-size: 14pt; line-height: 1.45; color: #000; }
h1 { font-size: 24pt; margin: 0 0 6pt 0; }
h2 { font-size: 18pt; margin: 10pt 0 6pt 0; }
h3 { font-size: 15pt; margin: 10pt 0 6pt 0; }
p { margin: 0 0 8pt 0; }
p.num { text-align: center; margin-top: 24pt; }
p.figure { color: #777; font-style: italic; }
p.picture { text-align: center; margin: 6pt 0 10pt 0; }
table { width: 100%; margin: 6pt 0 10pt 0; }
td, th { border: 1px solid #000; padding: 4pt 6pt; height: 34pt; font-size: 13pt;
         line-height: 1.2; vertical-align: middle; }
td.pic, th.pic { text-align: center; }
"""

# Tighter spacing for a worksheet that does not quite fit on one A4 page. It
# replaces the @page rule (xhtml2pdf rejects a second one) and overrides the rest.
_PAGE_RULE = "@page { size: a4 portrait; margin: 1.8cm 2cm; }"
_COMPACT_PAGE_RULE = "@page { size: a4 portrait; margin: 1.2cm 1.6cm; }"
COMPACT_CSS = """
body { font-size: 12.5pt; line-height: 1.3; }
h1 { font-size: 21pt; margin: 0 0 4pt 0; }
h2 { font-size: 16pt; margin: 6pt 0 4pt 0; }
h3 { font-size: 14pt; margin: 6pt 0 4pt 0; }
p { margin: 0 0 5pt 0; }
p.num { margin-top: 10pt; }
td, th { height: 28pt; font-size: 12pt; }
"""
_EMPTY_INLINE = re.compile(r"<(p|b|strong|u|i|em|div)>\s*</\1>")

MIN_BLANK_DOTS = 14  # models tend to shorten answer blanks; give students room to write

# Pictures
CONTENT_WIDTH_CM = 17.0      # A4 minus the left/right page margins
MAX_PICTURE_HEIGHT_CM = 18.0
_PICTURE_MARGIN = 0.03       # crop this much (of the box size) around the model's box
_MIN_PICTURE_PX = 24         # smaller crops are noise
_MAX_PICTURE_AREA = 0.85     # a "picture" covering most of the photo is a misread page
MAX_PICTURES_PER_PAGE = 12   # more is a misread page (or an attempt to bloat the PDF)
_MAX_TOTAL_PICTURE_AREA = 1.0  # all pictures together may not cover more than the photo
_PICTURE_DPI = 200           # crops are scaled down to this print resolution
_PICTURE_SPACING_CM = 0.9    # margins + line spacing around one picture block in a cell
_PLACEHOLDER = re.compile(r'<p class="figure" data-fig="(\d+)">\[picture\]</p>')
_CELL_MARK = re.compile(r' data-cell="(\d+)"')
_ROW_MARK = re.compile(r' data-row="(\d+)"')
_ROW_TAGS = re.compile(r"<(/?)(tr|td|th)\b([^>]*)>")
_PAPER_MIN_BRIGHTNESS = 0.65  # a crop border this bright (vs. the page paper) can be paper...
_PAPER_MAX_TINT = 1.15        # ...if it is no more tinted than this (channel ratio spread)


class RetypeError(RuntimeError):
    """No model in the chain could re-typeset the page."""

    def __init__(self, message: str, exhausted: Optional[Set[str]] = None):
        super().__init__(message)
        self.exhausted = exhausted or set()  # models that hit their daily quota meanwhile


@dataclass
class LineDiff:
    """A line on which the two readers disagree ("" = the line is missing in that reading)."""

    primary: str
    secondary: str


@dataclass
class Figure:
    """A picture on the worksheet, located by the model."""

    box: Optional[Tuple[int, int, int, int]]  # ymin, xmin, ymax, xmax on 0-1000; None if unusable
    has_handwriting: Optional[bool] = None    # student marks on it? None = the model did not say
    embedded: bool = False                    # copied into the page (else a placeholder is shown)
    cell: Optional[int] = None                # table cell holding it (its height must fit it)
    row: Optional[int] = None                 # table row holding that cell


@dataclass
class RetypeResult:
    html: str                       # sanitized body HTML of the blank worksheet
    model: str                      # model whose reading is used
    verify_model: Optional[str] = None
    flagged: List[LineDiff] = field(default_factory=list)
    handwriting: list = field(default_factory=list)   # what the model said it left out
    figures: List[Figure] = field(default_factory=list)  # pictures, in page order
    notes: List[str] = field(default_factory=list)    # fallbacks and skipped checks, for the user
    exhausted: Set[str] = field(default_factory=set)  # models that hit their quota (skip them next)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def generation_config(model: str) -> dict:
    cfg = {
        "temperature": 0,
        "responseMimeType": "application/json",
        "mediaResolution": "MEDIA_RESOLUTION_HIGH",
    }
    if not model.startswith("gemma"):  # Gemma rejects thinking settings (HTTP 400)
        cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
    return cfg


def parse_reply(text: str) -> Tuple[str, list]:
    """Returns (html, handwriting list) from the model's JSON reply (tolerant of fences/prose)."""
    t = re.sub(r"^\s*```(?:json|html)?\s*|\s*```\s*$", "", text.strip())
    data = None
    try:
        data = json.loads(t, strict=False)  # strict=False: raw newlines/tabs inside strings
    except json.JSONDecodeError:
        decoder = json.JSONDecoder(strict=False)
        for match in re.finditer(r"[{\[]", t):
            try:
                candidate, _ = decoder.raw_decode(t[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, (dict, list)):
                data = candidate
                break
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if isinstance(data, dict):
        html = data.get("html") or ""
        handwriting = data.get("handwriting") or []
    elif t.startswith("<"):
        html, handwriting = t, []  # plain HTML instead of JSON
    else:
        html, handwriting = "", []
    if not isinstance(html, str) or not html.strip():
        raise ValueError("the reply contained no worksheet HTML")
    return html, handwriting if isinstance(handwriting, list) else []


def _next(i: int, models: Sequence[str]) -> str:
    return "; trying the next model." if i < len(models) - 1 else "."


def _read_page(
    jpeg_bytes: bytes, api_key: str, models: Sequence[str], notes: List[str], exhausted: Set[str]
) -> Tuple[str, str, List[Figure], list, List[str]]:
    """
    Tries the models in order. Returns (model, sanitized html, pictures,
    handwriting, models after it). Raises RetypeError if every model fails.
    """
    first_note = len(notes)
    for i, model in enumerate(models):
        try:
            reply = call_gemini(
                [image_part(jpeg_bytes), {"text": RETYPE_PROMPT}],
                api_key,
                model,
                generation_config(model),
            )
            raw_html, handwriting = parse_reply(reply)
            html, figures = sanitize_html(raw_html)
            if not text_lines(html):
                raise ValueError("the reply contained no printed text")
            return model, html, figures, handwriting, list(models[i + 1:])
        except GeminiError as e:
            if e.is_quota and e.daily_quota:
                exhausted.add(model)
                notes.append(f"{model}: today's free quota is used up{_next(i, models)}")
            elif e.is_quota:
                notes.append(f"{model}: per-minute limit reached for this page{_next(i, models)}")
            elif e.status in (401, 403) or "API key not valid" in str(e):
                raise RetypeError(f"The Gemini API key was rejected: {e}", exhausted) from e
            else:
                notes.append(f"{model}: failed ({e}){_next(i, models)}")
        except ValueError as e:
            notes.append(f"{model}: unusable reply ({e}){_next(i, models)}")
    raise RetypeError("No model could read the page. " + " ".join(notes[first_note:]), exhausted)


def transcribe_page(
    jpeg_bytes: bytes,
    api_key: str,
    verify: bool = True,
    skip: Iterable[str] = (),
    chain: Sequence[str] = RETYPE_CHAIN,
) -> RetypeResult:
    """
    Reads one worksheet photo into sanitized HTML. With `verify`, a second,
    different model reads it too and disagreeing lines are reported.
    `skip` lists models known to be out of quota (from earlier pages).
    """
    notes: List[str] = []
    exhausted: Set[str] = set()
    skip = set(skip)
    models = [m for m in chain if m not in skip]
    if not models:
        raise RetypeError("Every model's free quota is used up for now. Try again later.")

    model, html, figures, handwriting, rest = _read_page(
        jpeg_bytes, api_key, models, notes, exhausted
    )
    result = RetypeResult(
        html=html, model=model, handwriting=handwriting, figures=figures, notes=notes,
        exhausted=exhausted,
    )
    if verify:
        _cross_check(result, jpeg_bytes, api_key, rest)
    # Pictures go in last: the cross-check compares text only.
    result.html = embed_figures(result.html, result.figures, jpeg_bytes)
    boxed = sum(1 for f in result.figures if f.box)
    if boxed > MAX_PICTURES_PER_PAGE:
        result.notes.append(
            f"The page has {boxed} pictures; only the first {MAX_PICTURES_PER_PAGE} were copied."
        )
    return result


def _cross_check(result: RetypeResult, jpeg_bytes: bytes, api_key: str, rest: List[str]) -> None:
    if not rest:
        result.notes.append("No second model was available, so the text was not double-checked.")
        return
    try:
        verify_model, other_html, _, _, _ = _read_page(
            jpeg_bytes, api_key, rest, result.notes, result.exhausted
        )
    except RetypeError:
        result.notes.append(
            "No second model could read the page, so the text was not double-checked."
        )
        return
    result.verify_model = verify_model
    result.flagged = compare_readings(result.html, other_html)


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------
def parse_box(value: str) -> Optional[Tuple[int, int, int, int]]:
    """A "ymin,xmin,ymax,xmax" box on 0-1000, or None if it is not a usable box."""
    m = re.fullmatch(r"\s*(\d{1,4})" + r"\s*,\s*(\d{1,4})" * 3 + r"\s*", value or "")
    if not m:
        return None
    ymin, xmin, ymax, xmax = (int(v) for v in m.groups())
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    if (ymax - ymin) * (xmax - xmin) > _MAX_PICTURE_AREA * 1_000_000:
        return None
    return ymin, xmin, ymax, xmax


def crop_picture(page: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Cuts the boxed picture (plus a small margin) out of the page photo."""
    h, w = page.shape[:2]
    ymin, xmin, ymax, xmax = box
    y1, y2 = ymin * h / 1000, ymax * h / 1000
    x1, x2 = xmin * w / 1000, xmax * w / 1000
    my, mx = (y2 - y1) * _PICTURE_MARGIN, (x2 - x1) * _PICTURE_MARGIN
    y1, y2 = max(0, int(y1 - my)), min(h, int(round(y2 + my)))
    x1, x2 = max(0, int(x1 - mx)), min(w, int(round(x2 + mx)))
    if y2 - y1 < _MIN_PICTURE_PX or x2 - x1 < _MIN_PICTURE_PX:
        return None
    return page[y1:y2, x1:x2].copy()


def page_paper_colour(page: np.ndarray) -> np.ndarray:
    """The page's paper colour: the bright end of a downscaled copy of the photo."""
    small = cv2.resize(page, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    return np.percentile(small.reshape(-1, 3).astype(np.float32), 90, axis=0)


def whiten_background(crop: np.ndarray, page_paper: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Scales the colours so the paper around the picture becomes white. The crop's
    border is the reference when it looks like paper (nearly as bright as the
    page's paper and not tinted), which follows local lighting such as a shadow.
    When the model's box cut into the picture, the border is the picture itself
    (darker, or coloured like a sky), so the page's paper colour is used instead
    and the picture is not washed out.
    """
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.float32)
    paper = np.percentile(border, 90, axis=0)
    if page_paper is not None:
        ratios = paper / np.maximum(np.asarray(page_paper, dtype=np.float32), 1.0)
        looks_like_paper = (ratios.min() >= _PAPER_MIN_BRIGHTNESS
                            and ratios.max() / max(ratios.min(), 1e-6) <= _PAPER_MAX_TINT)
        if not looks_like_paper:
            paper = np.asarray(page_paper, dtype=np.float32)
    paper = np.maximum(paper, 40.0)  # never blow up colours from a dark reference
    out = crop.astype(np.float32) * (255.0 / paper)
    out[out.min(axis=2) > 235] = 255  # near-white paper -> pure white for printing
    return np.clip(out, 0, 255).astype(np.uint8)


def picture_size_cm(shape: Tuple[int, ...], box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Printed (width, height): the picture's share of the photo width, kept on the page."""
    h, w = shape[:2]
    width = min(CONTENT_WIDTH_CM, max(3.0, (box[3] - box[1]) / 1000 * CONTENT_WIDTH_CM * 1.1))
    width = round(min(width, MAX_PICTURE_HEIGHT_CM * w / h), 1)
    return width, width * h / w


def picture_tag(crop: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    """An <img> with a data: URI, sized like the picture's share of the photo width."""
    h, w = crop.shape[:2]
    width, _ = picture_size_cm(crop.shape, box)
    print_px = int(width / 2.54 * _PICTURE_DPI)
    if w > print_px:  # no need to embed more pixels than the printer can use
        crop = cv2.resize(crop, (print_px, max(1, round(h * print_px / w))),
                          interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode the picture")
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
    return f'<p class="picture"><img src="{uri}" style="width: {width:.1f}cm"></p>'


def embed_figures(html: str, figures: List[Figure], jpeg_bytes: bytes) -> str:
    """
    Replaces each picture placeholder whose box is usable with the cropped
    picture, up to MAX_PICTURES_PER_PAGE pictures covering at most the photo's
    area in total (the model's reply is untrusted and must not bloat the PDF).
    """
    page = None
    if any(f.box for f in figures):
        page = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if page is None:
        return _ROW_MARK.sub("", _CELL_MARK.sub("", html))
    page_paper = page_paper_colour(page)
    budget = {"count": MAX_PICTURES_PER_PAGE, "area": _MAX_TOTAL_PICTURE_AREA * 1_000_000}
    cell_heights: dict = {}

    def replace(match: re.Match) -> str:
        index = int(match.group(1))
        figure = figures[index] if index < len(figures) else None
        if figure is None or figure.box is None:
            return match.group(0)
        ymin, xmin, ymax, xmax = figure.box
        area = (ymax - ymin) * (xmax - xmin)
        if budget["count"] <= 0 or area > budget["area"]:
            return match.group(0)
        crop = crop_picture(page, figure.box)
        if crop is None:
            return match.group(0)
        try:
            tag = picture_tag(whiten_background(crop, page_paper), figure.box)
        except ValueError:
            return match.group(0)
        budget["count"] -= 1
        budget["area"] -= area
        figure.embedded = True
        if figure.cell is not None:
            height = picture_size_cm(crop.shape, figure.box)[1]
            cell_heights[figure.cell] = cell_heights.get(figure.cell, 0.4) + height + _PICTURE_SPACING_CM
        return tag

    html = _PLACEHOLDER.sub(replace, html)
    # xhtml2pdf squeezes images into the fixed row height (and takes a row's
    # height from its last cell), so every cell of a row holding pictures gets
    # the height of its tallest picture cell.
    row_heights: dict = {}
    cell_rows = {f.cell: f.row for f in figures if f.cell is not None}
    for cell, height in cell_heights.items():
        row = cell_rows.get(cell)
        if row is not None:
            row_heights[row] = max(row_heights.get(row, 0.0), height)

    def size_cell(match: re.Match) -> str:  # a cell outside any <tr>
        height = cell_heights.get(int(match.group(1)))
        if height is None or cell_rows.get(int(match.group(1))) is not None:
            return ""
        return f' style="height: {min(height, MAX_PICTURE_HEIGHT_CM + 1):.1f}cm"'

    return _size_picture_rows(_CELL_MARK.sub(size_cell, html), row_heights)


def _size_picture_rows(html: str, row_heights: dict) -> str:
    """Gives the cells directly inside each marked <tr> its height; removes the marks."""
    out, pos, rows = [], 0, []  # rows: one entry per open <tr> (its height or None)
    for m in _ROW_TAGS.finditer(html):
        closing, tag, attrs = m.groups()
        out.append(html[pos:m.start()])
        pos = m.end()
        if tag == "tr" and closing:
            if rows:
                rows.pop()
        elif tag == "tr":
            mark = _ROW_MARK.search(attrs)
            rows.append(row_heights.get(int(mark.group(1))) if mark else None)
            attrs = _ROW_MARK.sub("", attrs)
        elif not closing and rows and rows[-1]:
            height = min(rows[-1], MAX_PICTURE_HEIGHT_CM + 1)
            attrs = re.sub(r' style="[^"]*"', "", attrs) + f' style="height: {height:.1f}cm"'
        out.append(f"<{closing}{tag}{attrs}>")
    out.append(html[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------
_BLOCK_END = re.compile(r"<(?:br\s*/?|/p|/h[1-6]|/div|/li|/tr)>", re.I)
_BLOCK_START = re.compile(r"<(?:p|h[1-6]|div|li|tr|table|ul|ol)\b[^>]*>", re.I)
_PAIR_SIMILARITY = 0.6  # lines this similar are shown side by side as one disagreement


def text_lines(html: str) -> List[str]:
    """Visible text of sanitized HTML, one entry per paragraph / heading / table row."""
    t = re.sub(r"</t[dh]>", " | ", html, flags=re.I)
    t = _BLOCK_END.sub("\n", t)
    t = _BLOCK_START.sub("\n", t)  # also split text that sits outside any paragraph
    t = re.sub(r"<[^>]+>", "", t)
    t = html_lib.unescape(t)
    lines = []
    for line in t.splitlines():
        line = re.sub(r"\s+", " ", line).strip().strip("|").strip()
        if line:
            lines.append(line)
    return lines


def _comparable(line: str) -> str:
    """Ignores spacing and the length of dotted blanks, keeps letters, case and punctuation."""
    line = unicodedata.normalize("NFC", line).replace("…", "...")
    line = re.sub(r"\.{2,}", "..", line)
    return re.sub(r"[\s|]+", "", line)


def _align_block(a: List[str], b: List[str], na: List[str], nb: List[str]) -> List[LineDiff]:
    """
    Pairs the lines of a changed block by similarity (keeping their order), so a
    misread line is shown next to its counterpart even when the other reading
    has an extra or a missing line around it.
    """
    diffs, j = [], 0
    for i, line in enumerate(a):
        best, best_ratio = None, _PAIR_SIMILARITY
        for k in range(j, len(b)):
            ratio = SequenceMatcher(None, na[i], nb[k], autojunk=False).ratio()
            if ratio >= best_ratio:
                best, best_ratio = k, ratio
        if best is None:
            diffs.append(LineDiff(line, ""))
            continue
        diffs.extend(LineDiff("", extra) for extra in b[j:best])
        diffs.append(LineDiff(line, b[best]))
        j = best + 1
    diffs.extend(LineDiff("", extra) for extra in b[j:])
    return diffs


def compare_readings(primary_html: str, secondary_html: str) -> List[LineDiff]:
    # Pictures are cropped from the photo, not read, so their placeholders are not compared.
    a = [x for x in text_lines(primary_html) if x != "[picture]"]
    b = [x for x in text_lines(secondary_html) if x != "[picture]"]
    na, nb = [_comparable(x) for x in a], [_comparable(x) for x in b]
    diffs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if tag != "equal":
            diffs.extend(_align_block(a[i1:i2], b[j1:j2], na[i1:i2], nb[j1:j2]))
    return diffs


# ---------------------------------------------------------------------------
# Sanitizing
# ---------------------------------------------------------------------------
_ALLOWED_TAGS = {
    "h1", "h2", "h3", "p", "b", "strong", "u", "i", "em", "br", "div",
    "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li",
}
_VOID_TAGS = {"br"}
_BLOCK_TAGS = {"h1", "h2", "h3", "p", "div", "table", "ul", "ol"}
_DROP_CONTENT_TAGS = {"script", "style", "title", "iframe", "object", "math", "template"}
_PICTURE_TAGS = {"img", "svg"}  # <picture>/<canvas> are wrappers: their <img> is the figure
_YES = ("yes", "y", "true", "1", "ja")
_NO = ("no", "n", "false", "0", "nein")
_ALLOWED_CLASSES = {"num"}
_DOT_RUN = re.compile(r"[.…]{2,}|…")
# Symbols the built-in PDF fonts cannot draw (they would print as filled squares).
_GLYPH_FALLBACKS = str.maketrans({
    "☐": "[   ]", "□": "[   ]", "▢": "[   ]", "☑": "[✓]", "☒": "[✗]",
    "○": "( )", "◯": "( )", "◦": "·", "☆": "*", "ẞ": "SS",
})


def _widen_blank(match: re.Match) -> str:
    run = match.group(0)
    dots = run.count(".") + 3 * run.count("…")
    return "." * max(dots, MIN_BLANK_DOTS) if dots >= 3 else run


class _Sanitizer(HTMLParser):
    """
    Rebuilds the model's HTML from an allowlist and keeps it well-formed:
    every allowed element is closed (stray end tags are ignored), so one
    page's unclosed table cannot swallow the next page in the combined PDF.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.figures: List[Figure] = []
        self._open: List[str] = []
        self._start_tags: List[Tuple[int, List[str]]] = []  # (index in out, attributes) per open tag
        self._cells = 0  # table cells that hold a picture
        self._rows = 0   # table rows that hold a picture
        self._skip_depth = 0

    def _push(self, tag: str, kept: List[str]) -> None:
        self.out.append(f"<{tag}{''.join(' ' + k for k in kept)}>")
        self._open.append(tag)
        self._start_tags.append((len(self.out) - 1, kept))

    def _close_until(self, tag: str) -> List[Tuple[str, List[str]]]:
        """Closes open elements up to and including `tag`; returns them, innermost first."""
        closed = []
        while self._open:
            top = self._open.pop()
            _, kept = self._start_tags.pop()
            self.out.append(f"</{top}>")
            closed.append((top, kept))
            if top == tag:
                break
        return closed

    def _add_picture(self, attrs) -> None:
        values = dict(attrs)
        flag = (values.get("data-handwriting") or "").strip().lower()
        self.figures.append(Figure(
            box=parse_box(values.get("data-box") or ""),
            has_handwriting=True if flag in _YES else False if flag in _NO else None,
        ))
        # The placeholder is a block: end an open paragraph around it and reopen
        # it (with its inline formatting) afterwards, as a browser would.
        reopen = self._close_until("p") if "p" in self._open else []
        # Only this module ever writes data-fig/data-cell/data-row; model attributes
        # are never copied. embed_figures sizes the marked cells and rows to fit.
        self.out.append(f'<p class="figure" data-fig="{len(self.figures) - 1}">[picture]</p>')
        figure = self.figures[-1]
        for tag, (index, kept) in zip(reversed(self._open), reversed(self._start_tags)):
            if figure.cell is None and tag in ("td", "th"):
                figure.cell = self._mark(tag, index, kept, "data-cell", "_cells", 'class="pic"')
            elif figure.cell is not None and tag == "tr":
                figure.row = self._mark(tag, index, kept, "data-row", "_rows")
                break
            elif tag == "table":
                break
        for tag, kept in reversed(reopen):
            self._push(tag, list(kept))

    def _mark(self, tag: str, index: int, kept: List[str], name: str, counter: str,
              *extra: str) -> int:
        """Adds (or reads) a numbered marker attribute on an already written start tag."""
        mark = next((k for k in kept if k.startswith(f"{name}=")), None)
        if mark is None:
            number = getattr(self, counter)
            setattr(self, counter, number + 1)
            mark = f'{name}="{number}"'
            kept.extend([*extra, mark])
            self.out[index] = f"<{tag}{''.join(' ' + k for k in kept)}>"
        return int(mark.split('"')[1])

    def _open_in_table(self, tag: str) -> bool:
        """Is `tag` open in the innermost table (not in a table around it)?"""
        for open_tag in reversed(self._open):
            if open_tag == tag:
                return True
            if open_tag == "table":
                return False
        return False

    def _close_implicitly(self, tag: str) -> None:
        """Closes elements that the new start tag ends, as browsers do."""
        if tag in ("td", "th") and self._open and self._open[-1] in ("td", "th"):
            self._close_until(self._open[-1])
        elif tag == "tr" and self._open_in_table("tr"):
            self._close_until("tr")
        elif tag == "li" and self._open and self._open[-1] == "li":
            self._close_until("li")
        if tag in _BLOCK_TAGS and "p" in self._open:
            self._close_until("p")

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _PICTURE_TAGS:
            self._add_picture(attrs)
            if tag == "svg":
                self._skip_depth += 1  # its drawing content is not text
            return
        if tag not in _ALLOWED_TAGS:
            return
        self._close_implicitly(tag)
        kept = []
        for name, value in attrs:
            value = value or ""
            if (name in ("colspan", "rowspan") and tag in ("td", "th")
                    and value.isascii() and value.isdigit() and len(value) <= 2):
                kept.append(f'{name}="{min(int(value), 20)}"')
            elif name == "class" and value in _ALLOWED_CLASSES:
                kept.append(f'class="{value}"')
        if tag in _VOID_TAGS:
            self.out.append(f"<{tag}>")
        else:
            self._push(tag, kept)

    def handle_startendtag(self, tag, attrs):
        if tag in _DROP_CONTENT_TAGS:
            return  # self-closed: there is no content to drop
        skipping = self._skip_depth
        self.handle_starttag(tag, attrs)
        if tag == "svg" and not skipping:
            self._skip_depth -= 1  # <svg/> has no content either
        elif self._open and self._open[-1] == tag:
            self._close_until(tag)  # e.g. <td/>

    def handle_endtag(self, tag):
        if tag in _DROP_CONTENT_TAGS or (tag == "svg" and self._skip_depth):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or tag not in self._open:
            return  # dropped tag, or a stray end tag
        self._close_until(tag)

    def handle_data(self, data):
        if not self._skip_depth:
            text = html_lib.escape(data.translate(_GLYPH_FALLBACKS), quote=False)
            self.out.append(_DOT_RUN.sub(_widen_blank, text))

    def finish(self) -> str:
        self.close()
        while self._open:
            self._close_until(self._open[-1])
        html = "".join(self.out)
        # Empty wrappers (e.g. left around a picture the model put in its own <p>)
        # only add blank space; remove them, innermost first.
        while True:
            shorter = _EMPTY_INLINE.sub("", html)
            if shorter == html:
                return html
            html = shorter


def sanitize_html(raw_html: str) -> Tuple[str, List[Figure]]:
    """
    Keeps only simple formatting tags (no attributes except colspan/rowspan and
    the page-number class), drops scripts, styles, links and remote resources,
    replaces pictures with a numbered placeholder and closes every open tag.
    Returns (html, pictures in placeholder order).
    Raises ValueError for HTML the parser cannot handle.
    """
    s = _Sanitizer()
    try:
        s.feed(raw_html)
        return s.finish(), s.figures
    except AssertionError as e:  # html.parser asserts on some malformed markup (e.g. "<![x[")
        raise ValueError(f"malformed HTML ({e})") from e


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _resource_policy():
    """xhtml2pdf may fetch nothing: no network, no local files (the HTML is untrusted)."""
    from xhtml2pdf.config.resources import ResourceAccessPolicy

    return ResourceAccessPolicy(allow_remote=False, base_dir=None, max_render_seconds=120)


def render_pdf(pages_html: Sequence[str], compact: bool = False) -> bytes:
    """
    Renders sanitized page bodies as one A4 PDF, each worksheet on a new page.
    `compact` uses tighter spacing. Raises RuntimeError if rendering fails.
    """
    from xhtml2pdf import pisa  # heavy import, only needed here

    css = PAGE_CSS
    if compact:
        css = css.replace(_PAGE_RULE, _COMPACT_PAGE_RULE) + COMPACT_CSS
    body = "<pdf:nextpage />".join(f"<div>{page}</div>" for page in pages_html)
    doc = (
        f'<html><head><meta charset="utf-8"><style>{css}</style></head>'
        f"<body>{body}</body></html>"
    )
    out = io.BytesIO()
    try:
        status = pisa.CreatePDF(doc, dest=out, encoding="utf-8", resource_policy=_resource_policy())
    except Exception as e:  # reportlab raises plain Exception subclasses (e.g. LayoutError)
        raise RuntimeError(f"PDF rendering failed: {e}") from e
    if status.err:
        raise RuntimeError(f"PDF rendering failed ({status.err} error(s)).")
    return out.getvalue()


def page_count(pdf: bytes) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf)
    try:
        return len(doc)
    finally:
        doc.close()


def render_page(page_html: str) -> bytes:
    """
    Renders one worksheet. If it spills onto a second sheet (often just the page
    number), it is rendered again with tighter spacing and the shorter result
    is kept. Raises RuntimeError if rendering fails.
    """
    pdf = render_pdf([page_html])
    if page_count(pdf) > 1:
        compact = render_pdf([page_html], compact=True)
        if page_count(compact) < page_count(pdf):
            return compact
    return pdf


def merge_pdfs(pdfs: Sequence[bytes]) -> bytes:
    """Joins already rendered PDFs, so one heavy page cannot fail the whole document."""
    import pypdfium2 as pdfium

    merged = pdfium.PdfDocument.new()
    sources = []
    try:
        for data in pdfs:
            source = pdfium.PdfDocument(data)
            sources.append(source)
            merged.import_pages(source)
        out = io.BytesIO()
        merged.save(out)
        return out.getvalue()
    finally:
        for source in sources:
            source.close()
        merged.close()


def render_preview(pdf_bytes: bytes, scale: float = 1.4) -> List[bytes]:
    """PNG images of the PDF's pages, for showing in the app."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    images = []
    try:
        for i in range(len(pdf)):
            buf = io.BytesIO()
            pdf[i].render(scale=scale).to_pil().save(buf, format="PNG")
            images.append(buf.getvalue())
    finally:
        pdf.close()
    return images
