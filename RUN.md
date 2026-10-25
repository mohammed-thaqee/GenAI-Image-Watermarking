# Run the training on Google Colab

This run trains the encoder and decoder for two tests:

1. The stamped 400×400 file decodes on its own.
2. The same file still decodes after brightness, contrast, JPEG, and a border crop of about 10%.

The finder for a stamp inside a larger canvas is a later run. It needs stamps from this training, pasted onto large images.

## Dataset

Use **MIRFLICKR-25k** (25,000 images). That is the collection the StegaStamp paper trained on.

The 1 million image collection is a different, much larger download. This project does not need it. 25,000 covers are enough for 140,000 training steps at batch size 4, because each step draws a random photo and a new random message.

Download MIRFLICKR-25k from the Leiden page: [https://press.liacs.nl/mirflickr/](https://press.liacs.nl/mirflickr/). Pick the 25,000 image zip, not the 1 million image zip.

The photos are not 400×400. Width and height vary. The training loader converts each file to RGB, fits it to 400×400, and scales pixels to `[0, 1]`.

Before uploading, move about 200 images into a separate folder named `mirflickr_holdout`. Train on the rest. The test notebook reads the holdout so it does not score photos the model just trained on.

## What to put on Drive

Upload the whole `own` folder so this path exists:

```
/content/drive/MyDrive/own/stegwatermark/
/content/drive/MyDrive/own/notebooks/01_train_objectives_1_and_3.ipynb
/content/drive/MyDrive/own/notebooks/02_test_objectives_1_and_3.ipynb
```

The notebooks import `stegwatermark`. Uploading only the `.ipynb` file is not enough.

Also put the data zip on Drive, or download it from the Colab session.

## Start a GPU runtime

1. Open [https://colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook**, and choose `notebooks/01_train_objectives_1_and_3.ipynb`.
   Or upload the `own` folder to Drive first, then **File → Open notebook → Google Drive**.
3. **Runtime → Change runtime type → T4 GPU → Save.**

## Training notebook

Run the cells from top to bottom.

**Libraries.** Installs `lpips`, OpenCV, Pillow, and `bchlib`. It leaves Colab's existing PyTorch in place.

**Drive and project folder.** Mounts Drive, then finds the directory that contains `stegwatermark/`. The printed `PROJECT_ROOT` should end in `own`. If it raises `FileNotFoundError`, the folder is not at `MyDrive/own`.

**Data and checkpoints.** Set `DATA_DIR` to the folder of training JPEGs. Copy the images onto the Colab disk so the loader is not reading every file from Drive:

```python
!mkdir -p /content/data
!unzip -q "/content/drive/MyDrive/mirflickr25k.zip" -d /content/data/mirflickr
```

Change the zip path to wherever you stored it. After unzipping, `DATA_DIR` must be a folder that contains the images. Subfolders are fine. The loader walks the whole tree.

`SAVE_DIR` is `/content/drive/MyDrive/watermark_checkpoints` when Drive mounted. Checkpoints written only under `/content` disappear when the runtime stops.

**Smoke test.** Runs 2 steps on random images. Wait until it prints `smoke test finished`. This checks the GPU, the install, and the checkpoint write. It does not train a real watermark.

If this cell runs out of memory, the T4 is too busy. **Runtime → Restart session**, then run the cells again. For the real run, change `batch_size` from 4 to 2.

**Real training.** Leave `RESUME = ""` for a new run. Then run the cell.

```python
RESUME = ""
config = TrainConfig(
    data_dir=DATA_DIR,
    save_dir=SAVE_DIR,
    resume=RESUME,
    synthetic=False,
    batch_size=4,
    num_steps=140000,
    device="cuda",
)
run_training(config)
```

`synthetic=False` is required. `True` ignores MIRFLICKR and trains on noise.

The log prints every 100 steps:

```
step 100  bit_acc 0.53  clean 0.61  secret ...
```

`clean` is bit accuracy on the stamped image with no extra damage. `bit_acc` is the same batch after blur, color, JPEG, and crop. `clean` should reach about **0.95** first. `bit_acc` climbs after that, as the distortions get stronger.

Files written to `SAVE_DIR`:

| File | Purpose |
|---|---|
| `latest.pt` | Newest weights, optimizer, and step |
| `step_005000.pt`, `step_010000.pt`, ... | A copy every 5,000 steps |
| `train_log.csv` | The numbers from the log lines |
| `previews/step_XXXXXX.png` | Cover, stamped image, and residual side by side |

140,000 steps is several hours. Colab will disconnect. That is expected.

## Resume after a disconnect

1. Reopen the notebook and set the runtime to a T4 GPU again.
2. Run the library cell and the Drive cell.
3. Set the resume path, then run the training cell:

```python
RESUME = "/content/drive/MyDrive/watermark_checkpoints/latest.pt"
```

The loss ramps and the crop schedule continue from the saved step. Starting again with `RESUME = ""` resets those schedules and overwrites `latest.pt`.

## Test notebook

Open `notebooks/02_test_objectives_1_and_3.ipynb` in the same way. Runtime type: T4 GPU.

Run the cells from the top. Set:

```python
CHECKPOINT = "/content/drive/MyDrive/watermark_checkpoints/latest.pt"
IMAGE_DIR = "/content/data/mirflickr_holdout"
MESSAGE = "Hello"
```

`IMAGE_DIR` is the 200 images you held out. The message must be at most 7 characters.

The table has one block per image:

| Row | Pass |
|---|---|
| `clean file` | `bch_ok` true, and `text` is `Hello` |
| `brightness +0.2`, `brightness -0.2` | `bch_ok` true |
| `contrast 0.6`, `contrast 1.4` | `bch_ok` true |
| `jpeg 80`, `jpeg 50` | `bch_ok` true |
| `crop 5%`, `crop 10%` | `bch_ok` true |
| `crop 15%` | Stress test. Training only ramps crops to 10%. |
| `unwatermarked cover` | `bch_ok` **false** |

The last row is a normal photo with no watermark. If that row passes, do not use the checkpoint as an AI / not-AI flag.

Stamped previews are written to `/content/stamped_preview/`. Copy any you want to keep onto Drive before the runtime stops.

## When to stop training early

Open `train_log.csv` or watch the log. A usable checkpoint for these two objectives has:

- `clean` around 0.95 or higher
- `bit_acc` clearly above 0.90 after the distortion ramps have finished (past roughly step 20,000)
- the test notebook passing the clean row and the 10% crop row
- the unwatermarked row failing the BCH check

The edge penalty only starts at step 80,000. Let the run reach that region before judging border quality. You do not have to wait for all 140,000 steps if the test table already passes and then stays stable for several checkpoints.
