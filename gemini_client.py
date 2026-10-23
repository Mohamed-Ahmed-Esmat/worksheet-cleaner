"""
Minimal Gemini API (REST) client shared by the engines.

Plain `requests` instead of the google-genai SDK, so there is no extra
dependency. Busy-model 5xx answers (common on the free tier) are retried with
a short backoff; every failure is raised as GeminiError carrying the HTTP
status, so callers can tell a quota problem (429) from an outage (5xx).
"""

import base64
import re
import time
from typing import List, Optional

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_DELAYS = (2, 5, None)  # seconds before each retry; None = last attempt
RETRYABLE_STATUSES = (500, 502, 503, 504)
_INVISIBLE = re.compile(r"[\s\u200b-\u200d\u2060\ufeff]")  # pasted keys often carry these


class GeminiError(RuntimeError):
    """A failed Gemini call. `status` is the HTTP status, or None for network errors."""

    def __init__(self, message: str, status: Optional[int] = None, daily_quota: bool = False):
        super().__init__(message)
        self.status = status
        self.daily_quota = daily_quota  # 429 because the per-day quota is used up

    @property
    def is_quota(self) -> bool:
        return self.status == 429

    @property
    def is_unavailable(self) -> bool:
        """Overloaded, down or unreachable: worth trying another model."""
        return self.status is None or self.status in RETRYABLE_STATUSES


def _is_daily_quota(error: dict) -> bool:
    """True if a 429 was caused by a per-day quota (QuotaFailure ids like "...PerDay...")."""
    for detail in error.get("details") or []:
        for violation in (detail or {}).get("violations") or []:
            if "PerDay" in str((violation or {}).get("quotaId", "")):
                return True
    return "per day" in str(error.get("message", "")).lower()


def _redact(error: Exception, key: str) -> str:
    return str(error).replace(key, "***")


def image_part(jpeg_bytes: bytes) -> dict:
    return {
        "inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg_bytes).decode()}
    }


def clean_api_key(api_key: str) -> str:
    """
    Strips whitespace/zero-width characters and rejects anything but visible ASCII
    (key formats vary: classic "AIza..." keys and newer dotted ones), which would
    otherwise break the HTTP header and leak the key into the error message.
    """
    key = _INVISIBLE.sub("", api_key or "")
    if not re.fullmatch(r"[\x21-\x7e]+", key):
        raise GeminiError(
            "The Gemini API key contains invalid characters. Copy it again from AI Studio.", 401
        )
    return key


def call_gemini(
    parts: List[dict],
    api_key: str,
    model: str,
    generation_config: Optional[dict] = None,
    timeout: int = 180,
) -> str:
    """Runs generateContent and returns the answer text (thought parts excluded)."""
    key = clean_api_key(api_key)
    payload = {"contents": [{"parts": parts}]}
    if generation_config:
        payload["generationConfig"] = generation_config

    for delay in RETRY_DELAYS:
        try:
            resp = requests.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": key},
                json=payload,
                timeout=timeout,
            )
        except requests.ConnectionError as e:  # includes ConnectTimeout: nothing was sent yet
            if delay is None:
                raise GeminiError(f"Could not reach the Gemini API: {_redact(e, key)}") from e
            time.sleep(delay)
            continue
        except requests.RequestException as e:  # e.g. ReadTimeout: a stalled model, don't wait again
            raise GeminiError(f"Gemini API request failed: {_redact(e, key)}") from e
        if resp.status_code not in RETRYABLE_STATUSES or delay is None:
            break
        time.sleep(delay)

    if resp.status_code != 200:
        try:
            error = resp.json().get("error", {})
            err = error.get("message", resp.text)
        except ValueError:
            error, err = {}, resp.text
        if resp.status_code == 429:
            raise GeminiError(
                f"Gemini rate limit reached (429): {err} "
                "Free-tier limits reset per minute and per day.",
                429,
                daily_quota=_is_daily_quota(error),
            )
        raise GeminiError(f"Gemini API error ({resp.status_code}): {err}", resp.status_code)

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates returned")
        raise GeminiError(f"Gemini returned no answer: {reason}", resp.status_code)
    parts_out = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts_out if not p.get("thought"))
