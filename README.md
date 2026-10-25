# Watermarking AI-generated images

This project hides a short identifier inside an image so software can later tell that the image was watermarked. The mark has to survive being passed around as a normal picture file: saved as-is, captured inside a larger screenshot, or changed by brightness and a small crop.

The design follows [StegaStamp](https://arxiv.org/abs/1904.05343) (Tancik, Mildenhall, Ng, CVPR 2020). An encoder adds a faint residual to a 400×400 RGB image. A decoder reads the hidden bits back. A separate finder locates the stamped image when it sits inside a larger picture.

The reference implementation this plan is based on is the TensorFlow 1.x StegaStamp repo. Training for this project is a PyTorch port of that same architecture and the hyperparameters in `scripts/base.sh`.

## Objectives

1. **Clean file.** The encoded image is decoded on its own, with no other processing.
2. **Stamp inside a larger image.** A 1000×900 region contains the full 400×400 watermarked image plus other background or artwork. A finder locates the stamp, warps that region back to 400×400, and the decoder reads the bits.
3. **Light edits, for an AI / not-AI flag.** Brightness, contrast, hue, JPEG, and similar changes still decode. A small crop that trims the borders still counts as watermarked. The hidden signal is concentrated toward the center, and training includes those border crops. A crop that removes the center is out of scope.

Objective 3 is a detection decision (watermark present or not). Objectives 1 and 2 also recover the identifier. A result is accepted when the error-correcting code checks out. When a small border crop breaks the code, the finder mask can still mark the image as watermarked. That fallback needs a threshold measured on ordinary, non-watermarked photos, so normal images are not labeled as AI.

## Pipeline

```
cover image + message bits
        │
        ▼
   encoder  →  residual
        │
        ▼
   stamped = clip(cover + residual)     saved as PNG
        │
        ▼
   finder (only when the stamp may be inside a larger image)
        │  segmentation mask → quadrilateral → warp to 400×400
        ▼
   decoder  →  bits  →  BCH check
        │
        ├─ code valid      → watermarked, identifier recovered
        └─ code invalid    → not watermarked
                            (border-crop case: finder mask may still flag it)
```

## Training setup

Train on a Colab T4 GPU. Store checkpoints on Google Drive, including the optimizer state and the global step. The schedules (loss weights, warp strength, edge penalty) depend on that step. Restoring weights without the step restarts every ramp at zero.

Colab disconnects are expected. Saving every 5–10k steps makes a restart a resume, not a new run. Copy MIRFLICKR onto the Colab disk for reading, and write checkpoints to Drive.

PyTorch is the training stack. The network, losses, and schedule stay the same as the original repo. The TensorFlow 1.15 environment is a fallback only: it needs an old CUDA stack and is brittle on current Colab.

### Encoder

Input is a 400×400 RGB image in `[0, 1]` and a 100-bit message.

- Subtract 0.5 from the image and the message.
- `Linear(100 → 7500)` + ReLU, reshape to `50×50×3`, upsample ×8.
- Concatenate with the image (6 channels) and run a U-Net: stride-2 channels 32, 32, 64, 128, 256, then skip-connected upsampling at 128, 64, 32, 32.
- A 1×1 convolution with no activation outputs a 3-channel residual.
- Stamped image = cover + residual.

### Decoder

- Spatial transformer: three stride-2 convolutions (32, 64, 128), a 128-unit dense layer, then a 6-parameter affine head. Weights start at zero and the bias starts at identity, so the first batches do not warp.
- Apply the affine with `affine_grid` and `grid_sample`.
- Convolution stack matching the repo (32, 32, 64, 64, 64, 128, 128), then `Linear → 512` and `Linear → 100`.
- Train with `BCEWithLogitsLoss`.

### Critic

Five-convolution network from the repo. The score is the mean of the final map. RMSprop at learning rate `1e-5`, gradient clip ±0.25, weight clip ±0.01.

### Geometry and distortions

These are applied during training only. Encoding at deployment is `clip(image + residual)`.

1. Random homography. Corners move by up to ±10% of the width, ramped over 10k steps. The encoder hides the message on the warped cover; the residual is warped back. Empty borders are filled white.
2. Blur with a random Gaussian or line kernel.
3. Gaussian noise, standard deviation up to 0.02.
4. Brightness, contrast in `[0.5, 1.5]`, hue, and desaturation.
5. Differentiable JPEG, quality sampled from 50 to 100, with cubic rounding near zero so gradients flow through quantization.

Border crops of about 5–15% and a strong edge penalty are included so a trim of the frame still leaves a readable center. The edge penalty starts at step 80k and ramps over 10k steps.

### Loss schedule

From `scripts/base.sh`, Adam at learning rate `1e-4`, batch size 4, 140k steps:

```
loss = 2 · YUV L2  +  1.5 · LPIPS  +  1.5 · message BCE  +  0.5 · critic
```

YUV channel weights are Y=1, U=100, V=100. LPIPS uses AlexNet.

- Steps 0–1500: message loss only.
- L2, LPIPS, and critic weights ramp linearly over 15k steps.
- Warp strength ramps over 10k steps.
- Edge-weighted L2 starts at step 80k and ramps over 10k steps.

Watch bit accuracy first. It should reach about 0.95 before the image losses become strong.

### Message format

100 bits: 7 UTF-8 characters (56 bits) + 40 BCH parity bits + 4 zero pad bits.

```python
import bchlib
bch = bchlib.BCH(5, prim_poly=137)   # confirm the constructor against the installed bchlib
data = bytearray("Hello".ljust(7), "utf-8")
packet = data + bch.encode(data)
bits = [int(b) for byte in packet for b in f"{byte:08b}"] + [0] * 4
```

On decode, take the first 96 rounded bits, split data and parity, and run `bch.decode`. Accept the image only when that decode succeeds.

## Finder

The decoder expects the stamp to fill a 400×400 frame. A 1000×900 snip that also contains background has to be handled by a finder first.

- Fine-tune a segmentation network (the paper uses BiSeNet) to mark pixels that belong to a stamp.
- Training images: encoded 400×400 stamps pasted at random positions and scales into larger photos, with mild rotation and perspective. The label is the binary mask of the pasted rectangle.
- Test time: convex hull of each mask, fit a quadrilateral, homography back to 400×400, then the decoder.

Train this after the encoder produces real stamps. The finder has to learn this project's residual.

## Data

**Encoder and decoder:** MIRFLICKR-25k, the set used in the paper.

MIRFLICKR images are original Flickr JPEGs. They are not 400×400. Width and height vary, and each photo is landscape, portrait, or square. A smaller release has the long side resized to about 500 pixels; the full release keeps the uploaded resolution.

Every training image is processed the way `get_img_batch` does:

- Convert to RGB.
- `ImageOps.fit` to 400×400 (scale so the image covers 400×400, then center-crop).
- Divide by 255 so pixels are in `[0, 1]`.

Images smaller than 400×400 are upscaled by that fit. That is expected.

**Finder canvases:** DIV2K, as in the paper. MIRFLICKR photos are often too small to serve as a 1000×900 background. Paste finished stamps onto these larger images to build the finder set.

## Acceptance tests

Use the same hidden message in every test.

| Test | Pass |
|---|---|
| Decode the stamped PNG unchanged | BCH succeeds |
| Paste the stamp into a larger image and take a 1000×900 crop that still contains all of it | Finder returns the region and BCH succeeds |
| Brightness and the other training distortions, on held-out images | Bit accuracy stays high and BCH succeeds |
| Crop roughly 10% off each edge | Finder or decoder still flags a watermark |

The paper's printer-and-camera number (about 98.7% bit accuracy) is a reference for the original StegaStamp setup. It is not the pass bar for this project.

## Training order

1. Port encoder, decoder, critic, warp, and distortion pipeline. Overfit a handful of images and confirm the bits come back.
2. Train 140k steps on MIRFLICKR. Checkpoint to Drive with the step counter.
3. Encode a stamp set. Train the finder by pasting those stamps into DIV2K.
4. Run the acceptance tests above.
