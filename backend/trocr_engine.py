import re
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

_trocr_singleton: Optional["TrOCREngine"] = None

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "BH"
}

INDIAN_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")


class TrOCREngine:
    """High-accuracy Vision Transformer (ViT) License Plate Recognizer.
    
    Uses Microsoft TrOCR (Vision Encoder Decoder) running in FP16 on CUDA.
    Includes automated HSRP blue-emblem stripping and multi-pass resolution.
    """

    def __init__(self, model_name: str = "microsoft/trocr-small-printed", device: Optional[str] = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[TrOCR] Loading Vision Transformer ({model_name}) on {self.device}...")
        t0 = time.time()
        
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        use_fp16 = self.device.startswith("cuda")
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()
        
        # Warm-up run to compile CUDA graph
        dummy = torch.zeros((1, 3, 384, 384), dtype=torch_dtype, device=self.device)
        with torch.no_grad():
            _ = self.model.generate(dummy, max_new_tokens=4)
            
        load_time = round(time.time() - t0, 2)
        print(f"[TrOCR] Loaded successfully in {load_time}s on {self.device} (FP16={use_fp16})")

    def _infer_single(self, crop: np.ndarray) -> Tuple[str, float]:
        """Internal single-pass inference on a normalized crop."""
        h, w = crop.shape[:2]
        if h < 32 or w < 96:
            scale = max(32 / max(1, h), 96 / max(1, w))
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        use_fp16 = self.device.startswith("cuda")
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        if use_fp16:
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=15,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate average token probability
        confidence = 0.85
        if outputs.scores:
            probs = []
            for score in outputs.scores:
                step_prob = torch.softmax(score, dim=-1).max().item()
                probs.append(step_prob)
            if probs:
                confidence = float(np.mean(probs))

        clean = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        return clean, round(confidence, 4)

    def recognize(self, plate_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text with smart multi-pass HSRP handling and syntax ranking.
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return "", 0.0

        h, w = plate_bgr.shape[:2]

        # Candidate 1: Whole crop
        text_full, conf_full = self._infer_single(plate_bgr)
        
        # Candidate 2: Strip HSRP blue IND emblem (leftmost 14%)
        text_hsrp, conf_hsrp = "", 0.0
        if w >= 70:
            hsrp_crop = plate_bgr[:, int(w * 0.14):]
            text_hsrp, conf_hsrp = self._infer_single(hsrp_crop)

        # Check which candidate matches Indian State prefix
        valid_full = any(text_full.startswith(st) for st in INDIAN_STATES)
        valid_hsrp = any(text_hsrp.startswith(st) for st in INDIAN_STATES)

        if valid_hsrp and not valid_full:
            return text_hsrp, max(conf_hsrp, 0.90)
        elif valid_full and not valid_hsrp:
            return text_full, max(conf_full, 0.90)
        elif valid_hsrp and valid_full:
            return (text_hsrp, conf_hsrp) if conf_hsrp >= conf_full else (text_full, conf_full)
        else:
            # Check if text_full contains an embedded state code after leading noise
            for st in INDIAN_STATES:
                idx = text_full.find(st)
                if 0 < idx <= 3 and len(text_full[idx:]) >= 7:
                    return text_full[idx:], conf_full

            return (text_hsrp, conf_hsrp) if conf_hsrp > conf_full else (text_full, conf_full)


def get_trocr_engine() -> TrOCREngine:
    global _trocr_singleton
    if _trocr_singleton is None:
        _trocr_singleton = TrOCREngine()
    return _trocr_singleton

