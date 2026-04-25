import re
from typing import Dict, List, Tuple

ALNUM_KEEP_RE = re.compile(r"[^A-Z0-9]")
SPACE_HYPHEN_RE = re.compile(r"[\s\-]+")
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")
INDIAN_PLATE_STRICT_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
}
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
}
DEFAULT_TEMPLATE = "LLDDLLDDDD"


def normalize_plate_text(raw_text: str) -> str:
    text = SPACE_HYPHEN_RE.sub("", raw_text.upper().strip())
    return ALNUM_KEEP_RE.sub("", text)


def apply_common_ocr_fixes(text: str) -> str:
    # Global cleanup pass for the most frequent OCR confusions.
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in text)


def is_indian_plate_like(text: str) -> bool:
    return bool(INDIAN_PLATE_RE.fullmatch(text))


def _force_char_type(char: str, expected: str) -> str:
    if expected == "L":
        if char.isalpha():
            return char
        return DIGIT_TO_LETTER.get(char, char)
    if expected == "D":
        if char.isdigit():
            return char
        return LETTER_TO_DIGIT.get(char, char)
    return char


def _apply_template(text: str, template: str) -> Tuple[str, int]:
    corrected = []
    edits = 0
    for char, token in zip(text, template):
        forced = _force_char_type(char, token)
        if forced != char:
            edits += 1
        corrected.append(forced)
    return "".join(corrected), edits


def _candidate_templates(text_len: int) -> List[str]:
    templates = []
    for district_len in (2, 1):
        for series_len in (3, 2, 1):
            template = "LL" + ("D" * district_len) + ("L" * series_len) + "DDDD"
            if len(template) == text_len:
                templates.append(template)
    if text_len == len(DEFAULT_TEMPLATE):
        templates.insert(0, DEFAULT_TEMPLATE)
    return templates


def enforce_indian_format(text: str) -> Tuple[str, bool]:
    if not text:
        return "", False
    if is_indian_plate_like(text):
        return text, True

    templates = _candidate_templates(len(text))
    if not templates:
        return text, False

    best = text
    best_match = False
    best_edits = 10**9
    for template in templates:
        corrected, edits = _apply_template(text, template)
        if is_indian_plate_like(corrected):
            if edits < best_edits:
                best = corrected
                best_match = True
                best_edits = edits

    # Force strict layout for 10-char strings even when regex does not fully match.
    if not best_match and len(text) == len(DEFAULT_TEMPLATE):
        forced, _ = _apply_template(text, DEFAULT_TEMPLATE)
        return forced, False
    return best, best_match


def select_best_plate_candidate(
    candidates: List[Dict],
    min_confidence: float = 0.0,
) -> Dict:
    if not candidates:
        return {
            "raw_text": "",
            "cleaned_text": "",
            "confidence": 0.0,
            "method_used": "none",
            "variant": "",
            "indian_format_like": False,
        }

    variant_weight = {
        "original": 0.02,
        "gray": 0.04,
        "bilateral": 0.06,
        "clahe": 0.08,
        "otsu": 0.09,
        "adaptive": 0.08,
        "sharpen": 0.09,
        "sharpen_otsu": 0.10,
    }

    text_buckets: Dict[str, Dict] = {}
    for cand in candidates:
        raw_text = str(cand.get("raw_text", "")).strip()
        conf = float(cand.get("confidence", 0.0))
        method_used = str(cand.get("method_used", "unknown"))
        variant = str(cand.get("variant", "original"))

        if conf < min_confidence:
            continue

        normalized = normalize_plate_text(raw_text)
        if len(normalized) < 4:
            continue

        globally_fixed = apply_common_ocr_fixes(normalized)
        corrected_from_normalized, format_match_normalized = enforce_indian_format(normalized)
        corrected_from_global, format_match_global = enforce_indian_format(globally_fixed)

        if format_match_global and not format_match_normalized:
            cleaned_text = corrected_from_global
            format_match = True
        else:
            cleaned_text = corrected_from_normalized if corrected_from_normalized else normalized
            format_match = format_match_normalized
        strict_bonus = 0.05 if INDIAN_PLATE_STRICT_RE.fullmatch(cleaned_text) else 0.0
        score = conf + (0.25 if format_match else 0.0) + strict_bonus + (0.01 * len(cleaned_text)) + variant_weight.get(
            variant, 0.0
        )

        bucket = text_buckets.get(cleaned_text)
        if bucket is None:
            text_buckets[cleaned_text] = {
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "confidence": conf,
                "method_used": method_used,
                "variant": variant,
                "indian_format_like": format_match,
                "score": score,
                "votes": 1,
            }
        else:
            bucket["votes"] += 1
            bucket["confidence"] = max(float(bucket["confidence"]), conf)
            bucket["score"] = max(float(bucket["score"]), score) + 0.05
            if conf >= float(bucket["confidence"]):
                bucket["raw_text"] = raw_text
                bucket["method_used"] = method_used
                bucket["variant"] = variant
                bucket["indian_format_like"] = format_match or bool(bucket["indian_format_like"])

    if not text_buckets:
        return {
            "raw_text": "",
            "cleaned_text": "",
            "confidence": 0.0,
            "method_used": "none",
            "variant": "",
            "indian_format_like": False,
        }

    best = max(
        text_buckets.values(),
        key=lambda row: (float(row["score"]) + 0.06 * int(row["votes"]), float(row["confidence"]), len(row["cleaned_text"])),
    )
    return {
        "raw_text": str(best["raw_text"]),
        "cleaned_text": str(best["cleaned_text"]),
        "confidence": round(float(best["confidence"]), 4),
        "method_used": str(best["method_used"]),
        "variant": str(best["variant"]),
        "indian_format_like": bool(best["indian_format_like"]),
    }
