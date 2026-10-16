"""Tests for the detect-then-erase engine. The OpenRouter call is always mocked."""

from unittest import mock

import cv2
import numpy as np
import pytest
import requests

import handwriting_eraser as he

PAPER = (205, 210, 215)  # slightly tinted, like a real photo (BGR)
PRINT = (25, 25, 25)
BLUE_PEN = (150, 60, 30)


def make_page() -> np.ndarray:
    """A 1000x800 'photo' with printed text on the left and right."""
    img = np.full((1000, 800, 3), PAPER, dtype=np.uint8)
    cv2.putText(img, "Das ist", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, PRINT, 3)
    cv2.putText(img, "Gitarre", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, PRINT, 3)
    cv2.putText(img, "der", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, PRINT, 3)
    return img


def ink_pixels(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return int((gray < 120).sum())


def blue_pixels(img: np.ndarray) -> int:
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return int(((b - r) > 60).sum())


def openrouter_response(content: str, status: int = 200) -> mock.Mock:
    resp = mock.Mock(status_code=status, text=content)
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ---------------------------------------------------------------------------
# parse_marks
# ---------------------------------------------------------------------------
def test_parse_marks_converts_normalized_boxes_to_pixels():
    text = '{"marks": [{"box_2d": [100, 250, 200, 500], "type": "answer"}]}'
    [m] = he.parse_marks(text, width=800, height=1000)
    assert (m.x1, m.y1, m.x2, m.y2, m.kind) == (200, 100, 400, 200, "answer")


def test_parse_marks_tolerates_malformed_json():
    # Real Flash-Lite output: an extra closing brace after "type".
    text = '{"marks": [{"box_2d": [\n 275, 187, 292, 237],\n "type": "answer"\n }\n },' \
           '{"box_2d": [300, 187, 313, 243], "type": "mark"}]}'
    marks = he.parse_marks(text, width=1000, height=1000)
    assert [m.kind for m in marks] == ["answer", "mark"]


def test_parse_marks_defaults_missing_type_to_mark_and_fixes_box_order():
    [m] = he.parse_marks('{"marks": [{"box_2d": [500, 600, 400, 300]}]}', 1000, 1000)
    assert m.kind == "mark"
    assert (m.x1, m.y1, m.x2, m.y2) == (300, 400, 600, 500)


def test_parse_marks_drops_empty_boxes_and_clamps_to_image():
    text = '{"marks": [{"box_2d": [10, 10, 10, 50]}, {"box_2d": [900, 900, 1200, 1100]}]}'
    [m] = he.parse_marks(text, width=100, height=100)
    assert (m.x2, m.y2) == (100, 100)


def test_parse_marks_returns_empty_for_no_marks():
    assert he.parse_marks('{"marks": []}', 100, 100) == []


# ---------------------------------------------------------------------------
# erase_marks
# ---------------------------------------------------------------------------
def test_answer_box_erases_handwriting_and_keeps_neighbouring_print():
    img = make_page()
    cv2.putText(img, "die", (250, 205), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, BLUE_PEN, 3)
    answer = he.Mark(240, 160, 420, 215, "answer")

    out = he.erase_marks(img, [answer], scan_look=False)

    assert blue_pixels(out) == 0
    assert ink_pixels(out, 240, 160, 420, 215) == 0
    # Printed words on both sides are untouched.
    assert ink_pixels(out, 30, 160, 230, 215) == ink_pixels(img, 30, 160, 230, 215)
    assert ink_pixels(out, 490, 160, 700, 215) == ink_pixels(img, 490, 160, 700, 215)


def test_answer_box_erases_pencil_too():
    img = make_page()
    cv2.putText(img, "das", (250, 205), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, (90, 90, 90), 3)
    out = he.erase_marks(img, [he.Mark(240, 160, 420, 215, "answer")], scan_look=False)
    assert ink_pixels(out, 240, 160, 420, 215) == 0


def test_answer_box_keeps_printed_table_border():
    img = make_page()
    cv2.line(img, (100, 300), (700, 300), PRINT, 2)  # long printed rule through the box
    cv2.putText(img, "kam", (300, 290), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, BLUE_PEN, 2)
    out = he.erase_marks(img, [he.Mark(290, 255, 420, 310, "answer")], scan_look=False)
    assert ink_pixels(out, 300, 298, 410, 303) == ink_pixels(img, 300, 298, 410, 303)


def test_mark_box_erases_circle_but_keeps_circled_print():
    img = make_page()
    before_word = ink_pixels(img, 300, 465, 365, 505)
    cv2.ellipse(img, (332, 487), (60, 35), 0, 0, 360, BLUE_PEN, 3)
    circle = he.Mark(270, 450, 395, 525, "mark")

    out = he.erase_marks(img, [circle], scan_look=False)

    assert blue_pixels(out) == 0
    assert ink_pixels(out, 300, 465, 365, 505) == before_word


def test_mark_box_leaves_neutral_ink_alone():
    img = make_page()
    cv2.ellipse(img, (332, 487), (60, 35), 0, 0, 360, PRINT, 3)  # black-pen circle
    out = he.erase_marks(img, [he.Mark(270, 450, 395, 525, "mark")], scan_look=False)
    assert np.array_equal(out, img)


def test_scan_look_turns_paper_white():
    out = he.erase_marks(make_page(), [], scan_look=True)
    assert out[900, 400].min() >= 250


def test_ink_outside_boxes_is_never_touched():
    img = make_page()
    cv2.putText(img, "note", (500, 800), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, BLUE_PEN, 3)
    out = he.erase_marks(img, [he.Mark(240, 160, 420, 215, "answer")], scan_look=False)
    assert np.array_equal(out[700:900], img[700:900])


# ---------------------------------------------------------------------------
# detect_and_erase (OpenRouter mocked)
# ---------------------------------------------------------------------------
@pytest.fixture
def page_jpeg() -> bytes:
    img = make_page()
    cv2.putText(img, "die", (250, 205), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, BLUE_PEN, 3)
    return cv2.imencode(".png", img)[1].tobytes()


def test_detect_and_erase_happy_path(page_jpeg):
    reply = '{"marks": [{"box_2d": [160, 300, 215, 525], "type": "answer"}]}'
    with mock.patch.object(he.requests, "post", return_value=openrouter_response(reply)) as post:
        cleaned, marks, msg = he.detect_and_erase(page_jpeg, " sk-test ", model="google/x")

    assert cleaned is not None and len(marks) == 1
    assert "Erased 1" in msg
    call = post.call_args
    assert call.kwargs["json"]["model"] == "google/x"
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_detect_and_erase_no_handwriting_returns_page_with_warning(page_jpeg):
    with mock.patch.object(he.requests, "post", return_value=openrouter_response('{"marks": []}')):
        cleaned, marks, msg = he.detect_and_erase(page_jpeg, "k")
    assert cleaned is not None and marks == []
    assert msg.startswith("Warning")


def test_detect_and_erase_reports_api_error(page_jpeg):
    resp = openrouter_response("", status=401)
    resp.json.return_value = {"error": {"message": "No auth credentials found"}}
    with mock.patch.object(he.requests, "post", return_value=resp):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "bad")
    assert cleaned is None
    assert "401" in msg and "No auth credentials" in msg


def test_detect_and_erase_reports_timeout(page_jpeg):
    with mock.patch.object(he.requests, "post", side_effect=requests.Timeout("slow")):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "k")
    assert cleaned is None and "Detection failed" in msg


def test_detect_and_erase_reports_non_box_reply(page_jpeg):
    with mock.patch.object(he.requests, "post", return_value=openrouter_response("I can't help")):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "k")
    assert cleaned is None and "did not return boxes" in msg


# ---------------------------------------------------------------------------
# Gemini API provider (mocked)
# ---------------------------------------------------------------------------
def gemini_response(text: str, status: int = 200, thought: str = "") -> mock.Mock:
    parts = [{"text": thought, "thought": True}] if thought else []
    parts.append({"text": text})
    resp = mock.Mock(status_code=status, text=text)
    resp.json.return_value = {"candidates": [{"content": {"parts": parts}}]}
    return resp


def test_gemini_provider_calls_gemini_api_with_bare_model_id(page_jpeg):
    reply = '{"marks": [{"box_2d": [160, 300, 215, 525], "type": "answer"}]}'
    with mock.patch.object(he.requests, "post", return_value=gemini_response(reply)) as post:
        cleaned, marks, _ = he.detect_and_erase(
            page_jpeg, " AIza-test ", model="google/gemini-3.5-flash-lite",
            provider=he.PROVIDER_GEMINI,
        )

    assert cleaned is not None and len(marks) == 1
    url = post.call_args.args[0]
    assert url.endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "AIza-test"}
    body = post.call_args.kwargs["json"]
    assert body["generationConfig"]["response_mime_type"] == "application/json"
    assert body["contents"][0]["parts"][1]["text"] == he.DETECTION_PROMPT


def test_gemini_provider_ignores_thought_parts(page_jpeg):
    reply = '{"marks": []}'
    resp = gemini_response(reply, thought='{"box_2d": [1, 2, 3, 4]}')
    with mock.patch.object(he.requests, "post", return_value=resp):
        assert he.detect_marks(page_jpeg, "k", provider=he.PROVIDER_GEMINI) == reply


def test_gemini_rate_limit_explains_free_tier(page_jpeg):
    resp = gemini_response("", status=429)
    resp.json.return_value = {"error": {"message": "Quota exceeded for requests per minute."}}
    with mock.patch.object(he.requests, "post", return_value=resp):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "k", provider=he.PROVIDER_GEMINI)
    assert cleaned is None
    assert "429" in msg and "Quota exceeded" in msg and "OpenRouter" in msg


def test_gemini_blocked_prompt_is_reported(page_jpeg):
    resp = mock.Mock(status_code=200, text="")
    resp.json.return_value = {"promptFeedback": {"blockReason": "SAFETY"}}
    with mock.patch.object(he.requests, "post", return_value=resp):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "k", provider=he.PROVIDER_GEMINI)
    assert cleaned is None and "SAFETY" in msg


def test_openrouter_provider_adds_google_prefix_to_bare_model_id(page_jpeg):
    with mock.patch.object(he.requests, "post", return_value=openrouter_response('{"marks": []}')) as post:
        he.detect_marks(page_jpeg, "k", model="gemini-3.5-flash-lite")
    assert post.call_args.kwargs["json"]["model"] == "google/gemini-3.5-flash-lite"


def test_gemini_retries_temporary_overload(page_jpeg):
    busy = gemini_response("", status=503)
    busy.json.return_value = {"error": {"message": "high demand"}}
    ok = gemini_response('{"marks": []}')
    with mock.patch.object(he.requests, "post", side_effect=[busy, ok]) as post, \
            mock.patch.object(he.time, "sleep") as sleep:
        assert he.detect_marks(page_jpeg, "k", provider=he.PROVIDER_GEMINI) == '{"marks": []}'
    assert post.call_count == 2 and sleep.call_count == 1


def test_gemini_gives_up_after_repeated_overload(page_jpeg):
    busy = gemini_response("", status=503)
    busy.json.return_value = {"error": {"message": "high demand"}}
    with mock.patch.object(he.requests, "post", return_value=busy) as post, \
            mock.patch.object(he.time, "sleep"):
        cleaned, _, msg = he.detect_and_erase(page_jpeg, "k", provider=he.PROVIDER_GEMINI)
    assert cleaned is None and "503" in msg
    assert post.call_count == len(he._GEMINI_RETRY_DELAYS)
