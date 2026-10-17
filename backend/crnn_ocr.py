import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS) + 1  # 37


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        return self.fc(recurrent)


class CRNN(nn.Module):
    def __init__(self, nc: int = 1, nclass: int = NUM_CLASSES, nh: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 16x80

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 8x40

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 4x40

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 2x40

            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),  # -> 1x39
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)
        out = self.rnn(conv)
        return out.permute(1, 0, 2)  # T x B x C


class CRNNRecognizer:
    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CRNN] = None
        self.available = False
        self.weights_path = weights_path or str(
            Path(__file__).resolve().parents[1] / "runs" / "ocr_crnn" / "crnn_plate.pth"
        )
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.weights_path):
            print(f"[CRNNRecognizer] Checkpoint not found at {self.weights_path}")
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            self.model = CRNN(nc=1, nclass=NUM_CLASSES, nh=128)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            acc = checkpoint.get("exact_acc", 0.0)
            print(f"[CRNNRecognizer] Loaded trained CRNN model (Exact Acc={acc:.1f}% on validation).")
        except Exception as exc:
            print(f"[CRNNRecognizer] Failed to load checkpoint: {exc}")
            self.available = False

    def _preprocess(self, crop_bgr: np.ndarray, width: int = 160, height: int = 32) -> torch.Tensor:
        if crop_bgr.ndim == 3:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_bgr

        # Apply CLAHE to sharpen character edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
        return tensor.to(self.device)

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.available or self.model is None or crop_bgr.size == 0:
            return "", 0.0

        h, w = crop_bgr.shape[:2]

        # Multi-line handling: if plate aspect ratio is square (<1.8), it is a 2-line stacked plate
        if float(w) / float(max(1, h)) < 1.8:
            mid = h // 2
            top_half = crop_bgr[: int(mid * 1.1), :]
            bottom_half = crop_bgr[int(mid * 0.9) :, :]
            t1, c1 = self._predict_single(top_half)
            t2, c2 = self._predict_single(bottom_half)
            combined = t1 + t2
            avg_conf = (c1 + c2) / 2.0 if (c1 and c2) else max(c1, c2)
            if combined:
                return combined, float(avg_conf)

        return self._predict_single(crop_bgr)

    def _predict_single(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        tensor = self._preprocess(crop_bgr)
        with torch.no_grad():
            preds = self.model(tensor)  # T x 1 x C
            probs = torch.softmax(preds, dim=2)  # T x 1 x C
            max_probs, pred_indices = probs.max(dim=2)  # T x 1

            seq = pred_indices[:, 0].tolist()
            conf_seq = max_probs[:, 0].tolist()

            chars = []
            confs = []
            prev = 0
            for idx, conf in zip(seq, conf_seq):
                if idx != 0 and idx != prev:
                    ch = IDX_TO_CHAR.get(idx, "")
                    if ch:
                        chars.append(ch)
                        confs.append(conf)
                prev = idx

            text = "".join(chars)
            mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
            return text, mean_conf


crnn_recognizer = CRNNRecognizer()

