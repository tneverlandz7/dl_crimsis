# Team 4 – CrisisMMD Multimodal Disaster Classification

Shared environment for the CrisisMMD coursework (Parts A, B and C). It provides data loading, the image-feature cache, a text vocabulary, environmental run logging, and a shared training loop, metrics and text encoders (see [Shared training and evaluation interface](#shared-training-and-evaluation-interface)). Each owner implements their model on their feature branch.

## Repository layout

```
.
├── data/CrisisMMD_v2.0/     Raw dataset (git-ignored)
├── features/                Cached ResNet18 features (git-ignored)
├── checkpoints/             Model weights (git-ignored)
├── logs/                    Training logs (git-ignored)
├── results/                 Metrics, predictions (committed)
├── figures/                 Plots (committed)
├── src/
│   ├── config.py            Paths, labels, brief-defined hyperparameters, energy assumptions
│   ├── data.py              Split loading (IDs as strings), label encoding, image-feature alignment
│   ├── text.py              Tokeniser and training-only vocabulary (64-token limit)
│   ├── utils.py             Seeding, device selection, parameter counting
│   ├── env_logger.py        Environmental run logger
│   ├── cache_features.py    Frozen ResNet18 feature extraction
│   ├── metrics.py           Shared evaluation: metrics, prediction CSVs, confusion matrices
│   ├── train.py             Shared training loop (Adam, early stopping on val macro-F1)
│   ├── aggregate.py         Mean ± std across seeds, median-run confusion matrices
│   ├── baseline.py          Majority-class baseline
│   ├── run_text.py          Text-only experiments (TextCNN / BiLSTM)
│   ├── run_image.py         Image-only experiments on cached ResNet18 features
│   ├── run_fusion.py        Feature-fusion experiments (TextCNN / BiLSTM swap)
│   └── models/
│       ├── text.py          TextCNN and BiLSTM encoders (swappable), text-only classifier
│       ├── image.py         Shared image encoder and image-only classifier
│       └── fusion.py        Shared feature-fusion classifier
├── env_tracker.csv          Environmental log: one row per run
└── requirements.txt
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows/Linux with an NVIDIA GPU, install the CUDA build of PyTorch **before** the requirements; plain `pip install torch` on Windows gives a CPU-only build. RTX 50-series (Blackwell) GPUs need CUDA 12.8 or newer:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

```bash
export CARBON_INTENSITY_G_PER_KWH=207.05   # default: UK DESNZ 2024 annual average
```

## Dataset

Everyone downloads CrisisMMD v2.0 from the official Hugging Face repository, [QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) (about 1.9 GB, CC BY-NC-SA 4.0):

```bash
hf download QCRI/CrisisMMD --repo-type dataset --local-dir data/CrisisMMD_v2.0
```

The brief requires the official **agreed-label** splits (Part A 9,601 / 1,573 / 1,534; Part B 6,126 / 998 / 955). The Hugging Face JSON splits are the all-label splits (13,608 / 2,237 / 2,237) and must not be used for Parts A and B. Add the agreed-label TSVs from the official CrisisMMD data-split release (brief reference [1]). The official site (`crisisnlp.qcri.org`) is offline, so use the Internet Archive copy of the official zip:

```bash
curl -L -o agreed.zip "http://web.archive.org/web/20250715040913id_/https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip"
unzip agreed.zip 'crisismmd_datasplit_agreed_label/*.tsv' -d data/CrisisMMD_v2.0
```

The all-label splits (needed for Part C3) are archived at the same path as `crisismmd_datasplit_all.zip` (snapshot `20250715063618`). An unauthenticated `hf download` can stop partway with a `401 Unauthorized` error; re-run the same command and it resumes.

The layout should then be:

```
data/CrisisMMD_v2.0/
├── data_image/                           from Hugging Face
├── crisismmd_datasplit_agreed_label/     task_{informative,humanitarian}_text_img_agreed_lab_{train,dev,test}.tsv
└── ...                                   remaining Hugging Face files
```

Image paths in the TSVs (`data_image/<event>/<date>/<id>.jpg`) match the Hugging Face layout. Set `CRISISMMD_ROOT` if the data lives elsewhere.

## Shared image-feature cache

Do not regenerate the cache. Download the shared file from the team drive into `features/` and verify it:

```bash
shasum -a 256 features/resnet18_imagenet1k_v1.pt
# c8443e3e840cb487bfe121ab3e7ce3f88934eb2c3ad643f4c9705cc000baeb68
chmod 444 features/resnet18_imagenet1k_v1.pt   # read-only
```

| Property | Value |
|---|---|
| Images | 18,082 (every image in CrisisMMD v2.0; covers all Part A, B and C splits) |
| Features | `float32 [18082, 512]`, ResNet18 `IMAGENET1K_V1`, eval mode, torchvision transforms |
| Invalid images | 0 |
| Size | 37.6 MB |
| Logged as | `FEAT-resnet18-all-final-01` in `env_tracker.csv` |

The file is a dict with `image_id` (list of strings), `features`, `invalid` and `weights`; load it with `src.data.load_image_features()`. It was generated with:

```bash
python -m src.cache_features --all-images --num-workers 4 --exp-id FEAT-resnet18-all-final-01
```

## Using the shared modules

```python
from src.data import load_split, encode_labels, load_image_features, align_image_features
from src.text import Vocab

train = load_split("informative", "train")
vocab = Vocab.build(train["tweet_text"])                    # training split only
x_text = vocab.encode_batch(train["tweet_text"])            # [N, 64]
y = encode_labels(train, "informative")
feats, index, invalid = load_image_features()
x_img, has_img = align_image_features(train, feats, index)  # [N, 512], [N]
```

## Shared training and evaluation interface

Use these for every learned model (text-only, image-only, fusion, Part B, Part C) so all comparisons use identical training, model selection and metrics. [src/run_text.py](src/run_text.py) is a complete worked example to copy for a new model.

**Model contract.** A model's `forward` takes a batch **dict** and returns logits `[B, num_classes]`. Keys are whatever you put into the loader; the conventions are `"text"` (token IDs `[B, 64]`, long), `"image"` (cached features `[B, 512]`, float) and `"label"` (`[B]`, long).

**Text encoders** ([src/models/text.py](src/models/text.py)). TextCNN and BiLSTM have the same interface and output size, so the fusion swap only changes the name:

```python
from src.models.text import build_text_encoder, TEXT_OUT_DIM   # TEXT_OUT_DIM = 300

encoder = build_text_encoder("textcnn", vocab_size=len(vocab), pad_id=vocab.pad_id)   # or "bilstm"
features = encoder(batch["text"])                               # [B, 300] = encoder.out_dim
```

| Encoder | Architecture | Output |
|---|---|---|
| `textcnn` | 128-d embedding → Conv1d with filter sizes 3/4/5 × 100 filters → ReLU → max-over-time | 300 |
| `bilstm` | 128-d embedding → one-layer BiLSTM, 150 per direction, padding ignored → [last fwd; last bwd] | 300 |

Text-only defaults: dropout 0.5 before the classifier, Adam lr 1e-3, vocabulary = training tokens seen at least twice (`min_freq=2`, 6,560 tokens for Part A).

**Feature fusion** ([src/models/fusion.py](src/models/fusion.py)). Both variants reuse the shared image branch and classifier; only the text encoder changes:

```text
cached ResNet18 [512] -> ImageEncoder [256] --+
                                                   +-> concatenate [556] -> Linear [128] -> ReLU -> Dropout -> classes
tweet tokens [64] -> TextCNN or BiLSTM [300] -----+
```

The default image and fusion dropout rates are 0.5. Keep the image branch, 128-unit fusion layer, optimiser and training budget unchanged for the TextCNN/BiLSTM comparison.

**Training and evaluation** ([src/train.py](src/train.py), [src/metrics.py](src/metrics.py)):

```python
from src.train import make_loader, train_model, predict, plot_learning_curves
from src.metrics import compute_metrics, save_predictions, save_json, result_dir
from src.utils import set_seed, count_trainable_params, get_device

set_seed(seed)                                                  # before building the model
train_loader = make_loader({"text": x_text, "image": x_img, "label": y}, shuffle=True, seed=seed)
dev_loader = make_loader({"text": x_text_dev, "image": x_img_dev, "label": y_dev})

result = train_model(model, train_loader, dev_loader, task, device)   # restores best-validation weights
# result.selected_epoch, .best_val_macro_f1, .epochs_run, .early_stopped, .train_time_s, .history

preds, probs, _ = predict(model, test_loader, device)
metrics = compute_metrics(y_test, preds, task)                  # accuracy, macro/weighted F1, per-class, confusion matrix
save_predictions(test_df, preds, probs, task, out_dir / "predictions_test.csv")
```

- `train_model` follows the brief: Adam, batch size 64, at most 20 epochs, checkpoint chosen by validation macro-F1, patience 3.
- **Part B imbalance:** pass `criterion=nn.CrossEntropyLoss(weight=w)` to `train_model`, or `sampler=WeightedRandomSampler(...)` to `make_loader`. Compute `w` from training labels only.
- Evaluate on **test only in final runs**. Debug and exploratory runs should use validation only.

**Results layout.** Final runs write to `results/<part>/<task>/<model>/seed<k>/`:

- `metrics.json` with at least `model`, `seed`, `experiment_id`, `val_macro_f1`, `selected_epoch`, `trainable_params`, `train_time_s`, `dev` and `test` (the last two from `compute_metrics`).
- `predictions_dev.csv`, `predictions_test.csv` and `learning_curve.png`.

`python -m src.aggregate --part A --task informative` then picks up every model in that folder and writes `summary.csv` (mean ± std over seeds), `runs.csv` (one row per run) and the test confusion matrix of each model's median-validation run in `figures/`. Name model folders distinctly, e.g. `image_only`, `fusion_textcnn`, `fusion_bilstm`.

**Commands:**

```bash
python -m src.baseline --task informative --exp-id A-inf-majority-s0-final-01
python -m src.run_text --task informative --encoder textcnn --seed 0 --run-type debug --exp-id A-inf-textcnn-s0-dbg-03
python -m src.run_text --task informative --encoder bilstm --seed 1 --run-type final --exp-id A-inf-bilstm-s1-final-01
python -m src.run_image --task informative --seed 0 --run-type final --exp-id A-inf-image-s0-final-01
python -m src.run_fusion --task informative --encoder textcnn --seed 0 --run-type debug --exp-id A-inf-fusion-textcnn-s0-dbg-01
python -m src.run_fusion --task informative --encoder textcnn --seed 0 --run-type final --exp-id A-inf-fusion-textcnn-s0-final-01
python -m src.run_fusion --task informative --encoder bilstm --seed 1 --run-type final --exp-id A-inf-fusion-bilstm-s1-final-01
python -m src.aggregate --part A --task informative
```

Every run is logged to `env_tracker.csv` automatically. Experiment IDs must be unique, so increase the final number for a repeat run.

The text-only, image-only and fusion runners use the same command pattern. Run each final learned-model variant with `--seed 0`, `--seed 1` and `--seed 2`, updating the `--exp-id` to match the model and seed. Some example IDs are already recorded; check `env_tracker.csv` and use a new final suffix for a repeat. All three runners select an available CUDA GPU automatically and otherwise use CPU; append `--device cpu` to run on CPU. Fusion uses the same shared defaults: Adam, learning rate `1e-3`, batch size 64, at most 20 epochs and early-stopping patience 3. Replace only `--encoder textcnn` with `--encoder bilstm` for the fusion architecture comparison.

**Current Part A results:** `results/A/informative/summary.csv` holds the majority baseline plus every learned model (3 seeds each). Test macro-F1 is `0.7899 +/- 0.0067` for text-only TextCNN, `0.7923 +/- 0.0072` for text-only BiLSTM, `0.8069 +/- 0.0071` for image-only, `0.8566 +/- 0.0078` for Fusion TextCNN, and `0.8597 +/- 0.0057` for Fusion BiLSTM.

**Fusion handoff:** [TextCNN fusion results](results/A/informative/fusion_textcnn/), [BiLSTM fusion results](results/A/informative/fusion_bilstm/) and the [six-case error analysis](results/A/informative/fusion_error_analysis.md) accompany the shared source and result formats. The recorded fusion runs used an NVIDIA GeForce RTX 4060 Laptop GPU, an Intel processor with 32 logical threads and approximately 17 GB RAM; cross-hardware training times should be interpreted with that context.

**Selected Part A fusion checkpoints:** provide one checkpoint for each fusion variant, following the existing per-model text/image handoff and experiment-ID filenames. Use each variant's median-validation run (the `cm_seed` in `summary.csv`), then retain that run's best-validation epoch. This keeps the delivered checkpoints aligned with the confusion-matrix runs.

| Model | Checkpoint | Seed | Selected epoch | Validation macro-F1 |
|---|---|---|---|---|
| Fusion TextCNN | `A-inf-fusion-textcnn-s0-final-01.pt` | 0 | 1 | 0.8411893418 |
| Fusion BiLSTM | `A-inf-fusion-bilstm-s0-final-01.pt` | 0 | 2 | 0.8478592615 |

The configurations and scores are in the matching [TextCNN metrics](results/A/informative/fusion_textcnn/seed0/metrics.json) and [BiLSTM metrics](results/A/informative/fusion_bilstm/seed0/metrics.json). Checkpoint selection uses validation scores only; the test-set mean and standard deviation still include all three seeds for each variant.

Both files are prepared for upload to the `checkpoints` subfolder of the [team Drive folder](https://drive.google.com/drive/folders/1XVTn41aTaLCrW2QymZhw8sJ1cst_LEBN); upload availability must be confirmed there. After downloading, place them in `checkpoints/`. Continue to reuse `features/resnet18_imagenet1k_v1.pt` from the shared cache resource and verify the cache checksum above. Checkpoints, feature caches and raw data remain outside Git, as for the text-only and image-only models.

## Environmental protocol

> **No code is merged into `main` unless every run behind it (debug, exploratory, final and failed) is logged in `env_tracker.csv` with runtime and energy.** Never delete rows.

### Confirm before running

Every run adds a permanent row to `env_tracker.csv`, so decide whether a run is needed **before** starting it.

- Before any command that trains, evaluates or extracts features (`src.baseline`, `src.run_text`, `src.cache_features` and any new run script), state which runs will be logged (experiment ID, run type, purpose) and confirm they are wanted.
- **AI coding assistants must ask the team member for explicit approval before executing any such command**, listing each experiment ID and noting that it will be written to `env_tracker.csv`. Never run verification, reproduction or "quick check" runs without that approval.
- Commands that only read existing results (`src.aggregate`, plotting from saved files) do not create rows and need no confirmation.

Wrap each run in `EnvRun`. It measures energy with CodeCarbon (falling back to the guidance §5.2 estimate), computes emissions, and appends a row on exit, including when the run crashes or is interrupted.

```python
from src.env_logger import EnvRun

with EnvRun("A-inf-textcnn-s0-dbg-01", part="A", task="informative", run_type="debug",
            model="textcnn", seed=0, batch_size=64) as run:
    ...  # training
    run.update(examples_processed=9601, epochs=12, val_macro_f1=0.78,
               run_status="early_stopped")
```

For runs outside Python code (e.g. a notebook cell), log manually:

```bash
python -m src.env_logger --experiment-id A-inf-textcnn-s0-dbg-02 --part A --task informative \
  --run-type debug --model textcnn --runtime-s 84 --run-status completed
```

- **Columns:** experiment ID, part, task, run type, model, seed, examples processed, epochs, batch size, device, runtime (s), energy (kWh), carbon intensity (gCO₂e/kWh), emissions (kgCO₂e), validation macro-F1, run status, cache/checkpoint used.
- **Experiment ID:** `<part>-<task>-<model>-s<seed>-<dbg|exp|final>-<nn>`. Must be unique.
- **run_type:** `debug`, `exploratory`, `final`, `failed`
- **run_status:** `completed`, `early_stopped`, `failed`, `interrupted`

## Git workflow

Feature-branch workflow. Each member works on their own branch and merges it into `main` when it is ready.

| Branch | Scope |
|---|---|
| `feature/part-a-baseline` | Majority-class baseline and metrics |
| `feature/part-a-textcnn` | Text-only models |
| `feature/part-a-image-only` | Image-only model on cached features |
| `feature/part-a-fusion` | Fusion model and TextCNN/BiLSTM swap |

Parts B and C follow the same pattern (`feature/part-b-<scope>`, `feature/part-c-<scope>`), with branches agreed when each phase starts.

```bash
git checkout main && git pull
git checkout -b feature/part-a-textcnn

git add src/ env_tracker.csv
git commit -m "Add TextCNN text-only model"

git push -u origin feature/part-a-textcnn

# merge into main when ready
git checkout main && git pull
git merge feature/part-a-textcnn
git push origin main
```

Merge locally with Git, not through the GitHub web interface: `env_tracker.csv` uses Git's union merge driver, so concurrent log rows combine automatically during a local merge.

**Before merging into `main`:**
- Every run from the branch is in `env_tracker.csv`.
- Vocabulary and any class weights are fitted on training data only.
- No data, feature caches or checkpoints are committed.


