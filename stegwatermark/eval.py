"""Encode a message and test objective 1 (clean file) and objective 3 (light edits)."""

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .codec import bits_to_text, text_to_bits
from .models import Decoder, Encoder


def load_encoder_decoder(checkpoint, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint, map_location=device)
    secret_size = int(payload.get("config", {}).get("secret_size", 100))
    encoder = Encoder(secret_size).to(device).eval()
    decoder = Decoder(secret_size).to(device).eval()
    encoder.load_state_dict(payload["encoder"])
    decoder.load_state_dict(payload["decoder"])
    return encoder, decoder, device, payload.get("step")


def load_cover(path, size=400):
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _bits_tensor(bits, device):
    return torch.tensor(bits, dtype=torch.float32, device=device).view(1, -1)


@torch.no_grad()
def embed(encoder, cover, bits):
    cover = cover.to(next(encoder.parameters()).device)
    residual = encoder(_bits_tensor(bits, cover.device), cover)
    stamped = (cover + residual).clamp(0, 1)
    return stamped, residual


@torch.no_grad()
def read_bits(decoder, image):
    image = image.to(next(decoder.parameters()).device)
    probabilities = torch.sigmoid(decoder(image))
    return probabilities[0].detach().cpu()


def bit_accuracy(probabilities, bits):
    predicted = (probabilities >= 0.5).float()
    target = torch.tensor(bits, dtype=torch.float32)
    return float((predicted == target).float().mean())


def _to_pil(image):
    array = (image[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(array)


def _from_pil(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def edit_brightness(image, amount):
    return (image + amount).clamp(0, 1)


def edit_contrast(image, factor):
    return ((image - 0.5) * factor + 0.5).clamp(0, 1)


def edit_jpeg(image, quality):
    buffer = io.BytesIO()
    _to_pil(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return _from_pil(Image.open(buffer))


def edit_border_crop(image, fraction):
    _, _, height, width = image.shape
    margin = int(round(fraction * height))
    if margin < 1:
        return image
    cropped = image[:, :, margin : height - margin, margin : width - margin]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def stamp_to_pil(stamped):
    return _to_pil(stamped)


def evaluate_cover(encoder, decoder, cover, message="Hello"):
    """Run the clean-file test and the light-edit tests on one cover tensor."""
    device = next(encoder.parameters()).device
    cover = cover.to(device)
    bits = text_to_bits(message, encoder.secret_size)
    stamped, _ = embed(encoder, cover, bits)

    cases = [
        ("clean file", stamped),
        ("brightness +0.2", edit_brightness(stamped, 0.2)),
        ("brightness -0.2", edit_brightness(stamped, -0.2)),
        ("contrast 0.6", edit_contrast(stamped, 0.6)),
        ("contrast 1.4", edit_contrast(stamped, 1.4)),
        ("jpeg 80", edit_jpeg(stamped, 80)),
        ("jpeg 50", edit_jpeg(stamped, 50)),
        ("crop 5%", edit_border_crop(stamped, 0.05)),
        ("crop 10%", edit_border_crop(stamped, 0.10)),
        ("crop 15%", edit_border_crop(stamped, 0.15)),
    ]
    rows = []
    for name, image in cases:
        probabilities = read_bits(decoder, image.to(device))
        text, ok = bits_to_text(probabilities.tolist())
        rows.append(
            {
                "edit": name,
                "bit_acc": round(bit_accuracy(probabilities, bits), 4),
                "bch_ok": ok,
                "text": text,
            }
        )

    # A cover with no residual should not pass the code check.
    raw = read_bits(decoder, cover)
    _, false_positive = bits_to_text(raw.tolist())
    rows.append(
        {
            "edit": "unwatermarked cover",
            "bit_acc": None,
            "bch_ok": false_positive,
            "text": None,
        }
    )
    return rows, stamped
