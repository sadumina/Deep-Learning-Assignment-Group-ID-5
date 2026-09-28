# Dataset Access: APTOS 2019 Blindness Detection

This project uses the **APTOS 2019 Blindness Detection** dataset: retinal fundus photographs graded for diabetic retinopathy (DR) severity. **The data is not stored in this repository.** You must obtain it yourself, after completing the verification steps below.

- **Source:** https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- **Provider:** Asia Pacific Tele-Ophthalmology Society (APTOS), distributed through Kaggle
- **Size:** about 9.5 GB compressed
- **Terms:** use is governed by the Kaggle competition rules. Read them before downloading.

## What the dataset contains

| File / folder | Description |
|---|---|
| `train.csv` | 3,662 rows: `id_code` (image name) and `diagnosis` (label 0-4) |
| `train_images/` | Labeled fundus images (`<id_code>.png`), varying resolutions |
| `test.csv` | Image IDs for the competition test set |
| `test_images/` | About 1,900 **unlabeled** images (no public ground truth) |
| `sample_submission.csv` | Kaggle submission format |

| Label | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The classes are heavily imbalanced (No DR is roughly half of the images). Because the competition test set has no public labels, this project creates its own labeled validation and test splits from `train.csv`.

## Step 1: Verification (required before download)

Complete all of the following before requesting the data:

1. **Kaggle account.** Register at https://www.kaggle.com and verify your email. Kaggle may also require **phone verification** before competition data can be downloaded (Settings -> Account -> Phone verification).
2. **Accept the competition rules.** Open https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules while logged in and click **I Understand and Accept**. Until you do, every download attempt fails with `403 Forbidden`, even with a valid API token.
3. **Face-to-face verification.** Access to the data for this project requires in-person verification before you obtain the dataset. `[Add who performs the verification, where/when, and what to bring, per your course requirements.]`

> Use your **own** Kaggle account. Do not share accounts or API tokens with teammates. Each person should accept the rules and generate their own token.

## Step 2: Download

Pick **one** of the options below.

### Option A: Kaggle API (recommended)

1. In Kaggle, go to **Settings -> API Tokens -> Create New Token**. Copy the token immediately, because it is shown only once.
2. Install the client and store the token:

```bash
pip install --upgrade kaggle
mkdir -p ~/.kaggle
echo "YOUR_API_TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

If you were given a legacy `kaggle.json` (`{"username": "...", "key": "..."}`) instead, place it at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.

3. Download and extract:

```bash
kaggle competitions download -c aptos2019-blindness-detection
unzip -q -o aptos2019-blindness-detection.zip -d data/
```

The `-o` flag overwrites existing files without prompting, which avoids the interactive "replace? [y]es, [n]o..." questions that hang notebook cells.

### Option B: Browser download

1. Complete Step 1, then open the **Data** tab of the competition page.
2. Click **Download All** and unzip into `data/`.

### Option C: Google Colab

Run in a notebook cell:

```python
!pip install -q --upgrade kaggle
!mkdir -p ~/.kaggle
!echo "YOUR_API_TOKEN" > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token
!kaggle competitions download -c aptos2019-blindness-detection
!unzip -q -o aptos2019-blindness-detection.zip -d /content/data/
```

Colab wipes its local disk when a session resets, so keep the preprocessed images and split files on Google Drive (`drive.mount('/content/drive')`) and copy them to local disk each session for faster training.

## Step 3: Verify the download

```python
import os, pandas as pd

data_dir = "data"   # adjust if different
df = pd.read_csv(f"{data_dir}/train.csv")

print(df.shape)                                       # expect (3662, 2)
print(df["diagnosis"].value_counts().sort_index())    # labels 0-4
print(len(os.listdir(f"{data_dir}/train_images")))    # expect 3662

missing = [i for i in df["id_code"] if not os.path.exists(f"{data_dir}/train_images/{i}.png")]
print("missing files:", len(missing))                 # expect 0
```

## Expected folder layout

```
data/
├── train.csv
├── train_images/
├── test.csv
├── test_images/
└── sample_submission.csv
```

Preprocessing (border crop, resize to 224x224, CLAHE) and the stratified train/val/test split are described in the main `README.md`. Preprocessed images are written to `processed_images/`, and split CSVs to `splits/`.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `401 Unauthorized` (even on `kaggle competitions list`) | The token is invalid, expired, or in the wrong format. Create a new token in Settings -> API Tokens and save it to `~/.kaggle/access_token`. Remove any stale `kaggle.json`. |
| `403 Forbidden` on download | The competition rules haven't been accepted for this account, or phone verification is missing. Complete Step 1. |
| `unzip: cannot find or open ...zip` | The download failed, so no zip exists. Fix the error above first. |
| `unzip` keeps asking to replace files | Re-run with `unzip -q -o` after clearing the folder (`rm -rf data`). |
| `FileNotFoundError: data/` | The data isn't in this session. On Colab, a reset deleted local files. Re-download, or use a copy stored on Drive. |
| Warning about an outdated `kaggle` version | Upgrade with `pip install --upgrade kaggle`. |

## Security and data handling

- **Never commit** API tokens, `kaggle.json`, `access_token`, the dataset, or preprocessed images. The repository `.gitignore` excludes them. If a token is ever exposed (chat, screenshot, commit), expire it immediately in Kaggle -> Settings -> API Tokens.
- Do not redistribute the images. Share this file and the Kaggle link so others can obtain the data themselves.
- The images are medical data. Use them only for this coursework and research, under the competition rules.

## Citation

Asia Pacific Tele-Ophthalmology Society (APTOS). *APTOS 2019 Blindness Detection.* Kaggle, 2019. https://www.kaggle.com/competitions/aptos2019-blindness-detection
