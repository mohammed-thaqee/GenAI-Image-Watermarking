"""Train the encoder and decoder for objectives 1 and 3.

Objective 1: the stamped 400×400 file decodes by itself.
Objective 3: brightness, color, JPEG, and a small border crop still decode.
The finder for a stamp sitting inside a larger canvas is a separate training run.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .data import CoverFolder, SyntheticCovers
from .distort import perspective_warp, photometric, random_border_crop, random_homographies, ramp, unwarp_white, warp_cover
from .models import Critic, Decoder, Encoder


@dataclass
class TrainConfig:
    data_dir: str = ""
    save_dir: str = "/content/drive/MyDrive/watermark_checkpoints"
    secret_size: int = 100
    batch_size: int = 4
    num_steps: int = 140000
    lr: float = 1e-4
    device: str = "cuda"
    synthetic: bool = False
    resume: str = ""
    log_every: int = 100
    save_every: int = 5000
    num_workers: int = 2

    # scripts/base.sh
    no_im_loss_steps: int = 1500
    l2_scale: float = 2.0
    l2_ramp: int = 15000
    lpips_scale: float = 1.5
    lpips_ramp: int = 15000
    secret_scale: float = 1.5
    g_scale: float = 0.5
    g_ramp: int = 15000
    y_scale: float = 1.0
    u_scale: float = 100.0
    v_scale: float = 100.0
    rnd_trans: float = 0.1
    trans_ramp: int = 10000
    edge_gain: float = 10.0
    edge_ramp: int = 10000
    edge_delay: int = 80000

    # train.py defaults that base.sh does not override
    bri: float = 0.3
    bri_ramp: int = 1000
    hue: float = 0.1
    hue_ramp: int = 1000
    sat: float = 1.0
    sat_ramp: int = 1000
    noise: float = 0.02
    noise_ramp: int = 1000
    contrast_low: float = 0.5
    contrast_high: float = 1.5
    contrast_ramp: int = 1000
    jpeg_quality: float = 50.0
    jpeg_ramp: int = 1000

    # Objective 3: trim up to 10% off each edge, after the message has started to stick.
    crop_max: float = 0.10
    crop_start: int = 1500
    crop_ramp: int = 20000


def _edge_falloff(size=400, speed=4):
    falloff = np.ones((size, size), np.float32)
    for i in range(int(size / speed)):
        weight = (np.cos(4 * np.pi * i / size + np.pi) + 1) / 2
        falloff[-i, :] *= weight
        falloff[i, :] *= weight
    for j in range(int(size / speed)):
        weight = (np.cos(4 * np.pi * j / size + np.pi) + 1) / 2
        falloff[:, -j] *= weight
        falloff[:, j] *= weight
    return torch.from_numpy(1.0 - falloff)


def _rgb_to_yuv(image):
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    u = -0.14714119 * red - 0.28886916 * green + 0.43601035 * blue
    v = 0.61497538 * red - 0.51496512 * green - 0.10001026 * blue
    return torch.cat([y, u, v], dim=1)


def yuv_loss(stamped, cover, falloff, gain, scales):
    diff = _rgb_to_yuv(stamped) - _rgb_to_yuv(cover)
    weight = 1.0 + gain * falloff.view(1, 1, *falloff.shape)
    diff = diff * weight
    per_channel = diff.square().mean(dim=(0, 2, 3))
    return torch.dot(per_channel, scales)


def _bit_accuracy(logits, secret):
    predicted = (logits.detach().sigmoid() >= 0.5).float()
    return (predicted == secret).float().mean().item()


def _save_preview(path, cover, stamped, residual):
    with torch.no_grad():
        residual_vis = (residual[:1].detach() * 5 + 0.5).clamp(0, 1)
        grid = torch.cat(
            [cover[:1].detach().clamp(0, 1), stamped[:1].detach().clamp(0, 1), residual_vis],
            dim=3,
        )
        array = (grid[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(array).save(path)


def _checkpoint(path, step, encoder, decoder, critic, opt_g, opt_d, config):
    torch.save(
        {
            "step": step,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "critic": critic.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def run_training(config: TrainConfig):
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, training on CPU. On Colab, switch the runtime to a GPU.")
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = save_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    if config.synthetic or not config.data_dir:
        dataset = SyntheticCovers()
        print("Using 64 synthetic images. Set data_dir to MIRFLICKR for a real run.")
        workers = 0
    else:
        dataset = CoverFolder(config.data_dir)
        print("covers: %d" % len(dataset))
        workers = config.num_workers
    if len(dataset) < config.batch_size:
        raise RuntimeError(
            "need at least %d images, found %d" % (config.batch_size, len(dataset))
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    encoder = Encoder(config.secret_size).to(device)
    decoder = Decoder(config.secret_size).to(device)
    critic = Critic().to(device)
    opt_g = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.lr)
    opt_d = torch.optim.RMSprop(critic.parameters(), lr=1e-5)

    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    for parameter in lpips_fn.parameters():
        parameter.requires_grad_(False)

    falloff = _edge_falloff().to(device)
    yuv_scales = torch.tensor([config.y_scale, config.u_scale, config.v_scale], device=device)
    start = 0
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        critic.load_state_dict(checkpoint["critic"])
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        start = int(checkpoint["step"]) + 1
        print("resumed at step %d" % start)

    log_path = save_dir / "train_log.csv"
    new_log = not log_path.exists() or start == 0
    log_file = log_path.open("a", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(
            ["step", "bit_acc", "bit_acc_clean", "secret", "yuv", "lpips", "g", "d", "crop_max"]
        )

    try:
        for step in range(start, config.num_steps):
            cover = next_batch().to(device, non_blocking=True)
            if cover.shape[-2:] != (400, 400):
                raise RuntimeError(
                    "covers must be 400×400 after loading (got %s). "
                    "CoverFolder resizes on the fly; check DATA_DIR and batch collation."
                    % (tuple(cover.shape),)
                )
            secret = torch.randint(0, 2, (cover.shape[0], config.secret_size), device=device, dtype=torch.float32)

            trans = min(config.rnd_trans * ramp(step, config.trans_ramp), config.rnd_trans)
            trans = random_fraction() * trans
            shift = int(np.floor(400 * trans))
            forward_np, inverse_np = random_homographies(cover.shape[0], 400, shift)
            forward = torch.from_numpy(forward_np).to(device=device, dtype=cover.dtype)
            inverse = torch.from_numpy(inverse_np).to(device=device, dtype=cover.dtype)

            warped, _ = warp_cover(cover, forward)
            residual_warped = encoder(secret, warped)
            encoded_warped = residual_warped + warped
            stamped = unwarp_white(encoded_warped, inverse)
            # The residual that would be saved is the unwarped one. Previews use it.
            residual = perspective_warp(residual_warped, inverse)

            distorted = photometric(stamped, step, config)
            crop_level = 0.0
            if step >= config.crop_start:
                crop_level = config.crop_max * ramp(step - config.crop_start, config.crop_ramp)
            distorted = random_border_crop(distorted, crop_level)
            logits = decoder(distorted)
            secret_loss = F.binary_cross_entropy_with_logits(logits, secret)

            warmup = step < config.no_im_loss_steps
            if warmup:
                opt_g.zero_grad(set_to_none=True)
                secret_loss.backward()
                opt_g.step()
                g_value = 0.0
                d_value = 0.0
                yuv_value = 0.0
                lpips_value = 0.0
            else:
                gain = 0.0
                if step > config.edge_delay:
                    gain = min(
                        config.edge_gain * (step - config.edge_delay) / float(config.edge_ramp),
                        config.edge_gain,
                    )
                image_loss = yuv_loss(stamped, cover, falloff, gain, yuv_scales)
                perceptual = lpips_fn(stamped * 2 - 1, cover * 2 - 1).mean()
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                fool = critic(encoded_warped)
                l2_w = min(config.l2_scale * step / float(config.l2_ramp), config.l2_scale)
                lpips_w = min(config.lpips_scale * step / float(config.lpips_ramp), config.lpips_scale)
                g_w = min(config.g_scale * step / float(config.g_ramp), config.g_scale)
                total = (
                    l2_w * image_loss
                    + lpips_w * perceptual
                    + config.secret_scale * secret_loss
                    + g_w * fool
                )
                opt_g.zero_grad(set_to_none=True)
                total.backward()
                opt_g.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

                d_loss = critic(warped.detach()) - critic(encoded_warped.detach())
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_value_(critic.parameters(), 0.25)
                opt_d.step()
                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-0.01, 0.01)
                g_value = float(fool.detach())
                d_value = float(d_loss.detach())
                yuv_value = float(image_loss.detach())
                lpips_value = float(perceptual.detach())

            if not math_isfinite(secret_loss):
                raise RuntimeError("secret loss became non-finite at step %d" % step)

            if step % config.log_every == 0 or step == config.num_steps - 1:
                with torch.no_grad():
                    clean_logits = decoder(stamped.clamp(0, 1))
                    clean_acc = _bit_accuracy(clean_logits, secret)
                acc = _bit_accuracy(logits, secret)
                row = [
                    step,
                    round(acc, 4),
                    round(clean_acc, 4),
                    round(float(secret_loss.detach()), 4),
                    round(yuv_value, 4),
                    round(lpips_value, 4),
                    round(g_value, 4),
                    round(d_value, 4),
                    round(crop_level, 4),
                ]
                writer.writerow(row)
                log_file.flush()
                print(
                    "step %d  bit_acc %.3f  clean %.3f  secret %.3f  yuv %.3f  lpips %.3f  crop %.3f"
                    % (step, acc, clean_acc, row[3], yuv_value, lpips_value, crop_level)
                )

            if step % config.save_every == 0 or step == config.num_steps - 1:
                _checkpoint(save_dir / ("step_%06d.pt" % step), step, encoder, decoder, critic, opt_g, opt_d, config)
                _checkpoint(save_dir / "latest.pt", step, encoder, decoder, critic, opt_g, opt_d, config)
                _save_preview(preview_dir / ("step_%06d.png" % step), cover, stamped, residual)
                print("saved %s" % (save_dir / "latest.pt"))
    finally:
        log_file.close()

    return str(save_dir / "latest.pt")


def random_fraction():
    return float(torch.rand(()))


def math_isfinite(value):
    return bool(torch.isfinite(value.detach()))
