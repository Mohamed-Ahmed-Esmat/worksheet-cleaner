"""
Re-typeset engine.

Instead of erasing ink from the photo, a free vision model reads only the
PRINTED text of the worksheet and returns it as simple HTML, which is then
rendered as a crisp A4 PDF. Handwriting, pen circles, shadows and watermarks
simply never make it into the output, and no printed letter is redrawn as
pixels.

The risk is the opposite one: a reader can silently misread a printed word
(typically one hidden under a student's circle). So each page is, by default,
read a second time by a different model and every line where the two readers
disagree is reported for the user to check.

Models are tried in order (RETYPE_CHAIN); a model that is out of quota,
overloaded or retired is skipped and the next one is used.
"""

import html as html_lib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Sequence, Set, Tuple

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
<img data-box="ymin,xmin,ymax,xmax"> where the box is normalized to 0-1000 on this photo.

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
table { width: 100%; margin: 6pt 0 10pt 0; }
td, th { border: 1px solid #000; padding: 4pt 6pt; height: 34pt; font-size: 13pt;
         line-height: 1.2; vertical-align: middle; }
"""

MIN_BLANK_DOTS = 14  # models tend to shorten answer blanks; give students room to write


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
class RetypeResult:
    html: str                       # sanitized body HTML of the blank worksheet
    model: str                      # model whose reading is used
    verify_model: Optional[str] = None
    flagged: List[LineDiff] = field(default_factory=list)
    handwriting: list = field(default_factory=list)   # what the model said it left out
    figures: List[str] = field(default_factory=list)  # data-box values of pictures (not rendered yet)
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
) -> Tuple[str, str, List[str], list, List[str]]:
    """
    Tries the models in order. Returns (model, sanitized html, picture boxes,
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
    if not verify:
        return result
    if not rest:
        notes.append("No second model was available, so the text was not double-checked.")
        return result
    try:
        verify_model, other_html, _, _, _ = _read_page(jpeg_bytes, api_key, rest, notes, exhausted)
    except RetypeError:
        notes.append("No second model could read the page, so the text was not double-checked.")
        return result
    result.verify_model = verify_model
    result.flagged = compare_readings(html, other_html)
    return result


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------
_BLOCK_END = re.compile(r"<(?:br\s*/?|/p|/h[1-6]|/div|/li|/tr)>", re.I)
_PAIR_SIMILARITY = 0.6  # lines this similar are shown side by side as one disagreement


def text_lines(html: str) -> List[str]:
    """Visible text of sanitized HTML, one entry per paragraph / heading / table row."""
    t = re.sub(r"</t[dh]>", " | ", html, flags=re.I)
    t = _BLOCK_END.sub("\n", t)
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
    a, b = text_lines(primary_html), text_lines(secondary_html)
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
_PICTURE_TAGS = {"img", "svg", "picture", "canvas"}
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
        self.figures: List[str] = []
        self._open: List[str] = []
        self._skip_depth = 0

    def _close_until(self, tag: str) -> None:
        while self._open:
            top = self._open.pop()
            self.out.append(f"</{top}>")
            if top == tag:
                return

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
            box = dict(attrs).get("data-box") or ""
            if re.fullmatch(r"\s*\d+(\s*,\s*\d+){3}\s*", box):
                self.figures.append(box.replace(" ", ""))
            else:
                self.figures.append("")
            self.out.append('<p class="figure">[picture]</p>')
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
        self.out.append(f"<{tag}{''.join(' ' + k for k in kept)}>")
        if tag not in _VOID_TAGS:
            self._open.append(tag)

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
            self.out.append(f"</{self._open.pop()}>")
        return "".join(self.out)


def sanitize_html(raw_html: str) -> Tuple[str, List[str]]:
    """
    Keeps only simple formatting tags (no attributes except colspan/rowspan and
    the page-number class), drops scripts, styles, links and remote resources,
    replaces pictures with a placeholder and closes every open tag.
    Returns (html, picture boxes; "" for a picture without a box).
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


def render_pdf(pages_html: Sequence[str]) -> bytes:
    """
    Renders sanitized page bodies as one A4 PDF, each worksheet on a new page.
    Raises RuntimeError if rendering fails.
    """
    from xhtml2pdf import pisa  # heavy import, only needed here

    body = "<pdf:nextpage />".join(f"<div>{page}</div>" for page in pages_html)
    doc = (
        f'<html><head><meta charset="utf-8"><style>{PAGE_CSS}</style></head>'
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
