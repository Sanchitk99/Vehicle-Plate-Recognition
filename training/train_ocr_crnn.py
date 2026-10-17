import os
import random
import string
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import DataLoader, Dataset

# Character set: blank=0, digits=1..10, letters=11..36
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS) + 1  # 37 (index 0 is CTC blank)

INDIAN_STATES = [
    "DL", "HR", "UP", "MH", "KA", "TN", "GJ", "RJ", "PB", "CH", "WB",
    "AP", "TS", "KL", "MP", "OD", "JH", "BR", "UK", "HP", "JK", "GA", "BH"
]

FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\impact.ttf",
]
VALID_FONTS = [f for f in FONTS if os.path.exists(f)]
if not VALID_FONTS:
    VALID_FONTS = ["arial.ttf"]


def generate_random_plate_text() -> str:
    """Generate realistic Indian and standard license plate strings."""
    state = random.choice(INDIAN_STATES)
    rto = f"{random.randint(1, 99):02d}"
    
    choice = random.random()
    if choice < 0.60:
        # Standard: e.g. DL01AB1234
        series = "".join(random.choices(string.ascii_uppercase, k=random.choice([1, 2])))
        num = f"{random.randint(1, 9999):04d}"
        return f"{state}{rto}{series}{num}"
    elif choice < 0.85:
        # Short / commercial / vintage: e.g. HR862415
        series = "".join(random.choices(string.ascii_uppercase, k=random.choice([0, 1])))
        num = f"{random.randint(1, 9999):04d}"
        return f"{state}{rto}{series}{num}"
    else:
        # General alphanumeric: 7 to 10 chars
        length = random.randint(7, 10)
        return "".join(random.choices(CHARACTERS, k=length))


def render_synthetic_plate(text: str, width: int = 160, height: int = 32) -> np.ndarray:
    """Generate a realistic synthetic plate crop with augmentations."""
    # Base background: white, light yellow, or light gray
    bg_color = random.choice([
        (255, 255, 255), (250, 250, 245), (240, 240, 235),
        (255, 230, 0),   # Commercial yellow plate
        (245, 245, 245),
    ])
    text_color = random.choice([(0, 0, 0), (15, 15, 15), (25, 25, 30)])

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Optional HSRP blue strip on the left (25% chance)
    if random.random() < 0.35:
        strip_w = random.randint(8, 14)
        draw.rectangle([0, 0, strip_w, height], fill=(0, 60, 160))
        text_x_start = strip_w + random.randint(3, 6)
    else:
        text_x_start = random.randint(3, 8)

    # Optional plate border
    if random.random() < 0.40:
        draw.rectangle([1, 1, width - 2, height - 2], outline=(30, 30, 30), width=1)

    # Pick a random font
    font_path = random.choice(VALID_FONTS)
    font_size = random.randint(16, 21)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Draw text with slight spacing
    # Format text into spaced display if it's an Indian plate (e.g. DL 01 AB 1234)
    if random.random() < 0.50 and len(text) >= 8:
        display_text = f"{text[:2]} {text[2:4]} {text[4:-4]} {text[-4:]}".replace("  ", " ").strip()
    else:
        display_text = text

    # Center vertically
    draw.text((text_x_start, random.randint(3, 6)), display_text, fill=text_color, font=font)

    # Add realistic noise / blur / lighting
    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    arr = np.array(img)

    # Random slight contrast / brightness variation
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
    arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)

    # Grayscale conversion
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Resize to exact model dimensions
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return gray


class SyntheticPlateDataset(Dataset):
    def __init__(self, size: int = 15000, width: int = 160, height: int = 32):
        self.size = size
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = generate_random_plate_text()
        img_gray = render_synthetic_plate(text, width=self.width, height=self.height)

        # Normalize to [-1.0, 1.0]
        tensor = torch.from_numpy(img_gray).float().unsqueeze(0) / 127.5 - 1.0

        target = [CHAR_TO_IDX[c] for c in text]
        target_tensor = torch.tensor(target, dtype=torch.long)
        return tensor, target_tensor, len(target)


def pad_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    flat_targets = torch.cat(targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, flat_targets, target_lengths


# ---------------------------------------------------------------------------
# CRNN Model Definition
# ---------------------------------------------------------------------------
class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        output = self.fc(recurrent)
        return output


class CRNN(nn.Module):
    def __init__(self, nc: int = 1, nclass: int = NUM_CLASSES, nh: int = 128):
        super().__init__()
        # CNN Backbone: 32x160 -> 1x39
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
        # conv features: B x 512 x 1 x T
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"
        # Reshape to B x T x C
        conv = conv.squeeze(2).permute(0, 2, 1)
        # RNN output: B x T x num_classes
        out = self.rnn(conv)
        # Log-softmax for CTC loss: T x B x num_classes
        out = out.permute(1, 0, 2).log_softmax(2)
        return out


def decode_predictions(pred_indices: torch.Tensor) -> List[str]:
    """CTC greedy decode: collapse adjacent repeats and remove blanks (0)."""
    # pred_indices: T x B
    t, b = pred_indices.size()
    results = []
    for batch_idx in range(b):
        seq = pred_indices[:, batch_idx].tolist()
        collapsed = []
        prev = 0
        for idx in seq:
            if idx != 0 and idx != prev:
                collapsed.append(IDX_TO_CHAR.get(idx, ""))
            prev = idx
        results.append("".join(collapsed))
    return results


def train_crnn(epochs: int = 12, batch_size: int = 64, lr: float = 0.001, save_dir: str = "runs/ocr_crnn"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[CRNN Training] Using device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_ckpt = save_path / "crnn_plate.pth"

    model = CRNN(nc=1, nclass=NUM_CLASSES, nh=128).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_dataset = SyntheticPlateDataset(size=12000, width=160, height=32)
    val_dataset = SyntheticPlateDataset(size=1000, width=160, height=32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
    )

    print(f"[CRNN Training] Total Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")
    start_time = time.time()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            preds = model(images)  # T x B x C
            t, b, c = preds.size()
            input_lengths = torch.full((b,), t, dtype=torch.long, device=device)

            loss = criterion(preds, targets, input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        exact_matches = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                preds = model(images)
                t, b, c = preds.size()
                input_lengths = torch.full((b,), t, dtype=torch.long, device=device)

                loss = criterion(preds, targets, input_lengths, target_lengths)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += float(loss.item())
                    val_batches += 1

                # Accuracy evaluation
                pred_indices = preds.argmax(2)  # T x B
                pred_texts = decode_predictions(pred_indices)

                # Reconstruct ground truth strings
                gt_idx = 0
                for length in target_lengths.tolist():
                    gt_chars = "".join([IDX_TO_CHAR[int(targets[gt_idx + k])] for k in range(length)])
                    gt_idx += length
                    if pred_texts[total_val_samples % b] == gt_chars:
                        exact_matches += 1
                    total_val_samples += 1

        avg_val_loss = val_loss / max(1, val_batches)
        exact_acc = float(exact_matches / max(1, total_val_samples)) * 100.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Exact Match Acc: {exact_acc:.1f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_val_loss < best_val_loss or epoch == epochs:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "characters": CHARACTERS,
                    "val_loss": avg_val_loss,
                    "exact_acc": exact_acc,
                    "epoch": epoch,
                },
                model_ckpt,
            )

    elapsed = time.time() - start_time
    print(f"\n[CRNN Training] Completed in {elapsed:.1f}s. Model checkpoint saved to: {model_ckpt}")
    return model_ckpt


if __name__ == "__main__":
    train_crnn(epochs=12, batch_size=64, lr=0.001)

