"""
ocr_module.py — VocalEyes OCR text reader with improved preprocessing
"""

import pytesseract
import cv2
import re
import time

_last_text = ""
_last_time = 0.0
_COOLDOWN = 4.0   # seconds between repeated reads
_MIN_LEN = 4      # minimum text length to speak


def _preprocess(frame):
    """Convert frame to clean grayscale for better OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize upward if small
    h, w = gray.shape
    if w < 640:
        scale = 640 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sharpen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    gray = cv2.dilate(gray, kernel, iterations=1)

    # Adaptive threshold
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return gray


def _clean(text: str) -> str:
    """Remove noise characters and normalize whitespace."""
    text = re.sub(r'[^A-Za-z0-9 \-.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def read_text(frame) -> str | None:
    """
    Attempt OCR on the given frame.
    Returns cleaned text if new and meaningful, else None.
    """
    global _last_text, _last_time

    processed = _preprocess(frame)

    try:
        raw = pytesseract.image_to_string(processed, config="--psm 6 --oem 3")
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return None

    text = _clean(raw)

    if len(text) < _MIN_LEN:
        return None

    if text == _last_text:
        return None

    if time.time() - _last_time < _COOLDOWN:
        return None

    _last_text = text
    _last_time = time.time()

    return text


def reset():
    """Reset OCR state."""
    global _last_text, _last_time
    _last_text = ""
    _last_time = 0.0
