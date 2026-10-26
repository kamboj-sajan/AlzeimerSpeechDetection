# Deeply-Fused Multi-Modal Co-Attention (AD from Speech)

A multimodal deep-learning pipeline for detecting Alzheimer's disease and dementia from spontaneous speech. It fuses acoustic embeddings (Wav2Vec2), transcript embeddings (BERT) and hand-crafted prosodic/paralinguistic features through stacked co-attention layers, and classifies each speaker as **Control (0)** or **Dementia (1)**.

---

## Architecture

![Architecture Diagram](architecture.png)

1. **Acoustic stream** — Wav2Vec2 (`facebook/wav2vec2-base-960h`) embeddings from the raw audio, fused with 6 prosodic features extracted with librosa (pitch mean/std, jitter, energy mean/std, spectral centroid).
2. **Linguistic stream** — BERT (`bert-base-uncased`) embeddings from the transcript, fused with 4 paralinguistic features extracted with spaCy (word count, type-token ratio, pause rate, average sentence length).
3. **Fusion** — a Transformer encoder layer per stream, then 3 stacked co-attention layers in which text attends to audio and audio attends to text.
4. **Classifier** — pooled representations of both streams are concatenated and passed to an MLP.

The pretrained backbones start frozen; the last 4 layers of each are unfrozen from epoch 3.

---

## Repository Layout

- `scriptsUpdates/` — current pipeline
  - `enhanced_config.py` — paths and hyperparameters
  - `enhanced_preprocess.py` — parses DementiaBank CHAT (`.cha`) transcripts and builds the metadata CSV
  - `process2_preprocess.py` — builds the metadata CSV for the PROCESS-2 corpus
  - `enhanced_dataset.py` — dataset, audio loading and cached hand-crafted feature extraction
  - `enhanced_model.py` — fusion + co-attention model
  - `enhanced_train.py` — 10-fold cross-validation training
  - `enhanced_results.py` — aggregated metrics, confusion matrix and training-curve plots
- `scripts/` — original baseline (Wav2Vec2 + BERT co-attention only, single train/validation split)
- `architecture.png`, `architecture-code.xml` — architecture diagram (draw.io source)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A CUDA GPU is strongly recommended (e.g. a free Kaggle or Google Colab T4). Training uses mixed precision automatically when CUDA is available.

---

## Data

The data is **not included** in this repository. Do not commit or redistribute it. Two corpora are supported:

### PROCESS-2

[PROCESS-2](https://huggingface.co/datasets/CognoSpeak/PROCESS-2) (gated; request access on Hugging Face) contains English recordings and transcripts from 400 participants (healthy, MCI, dementia) with a predefined train/test split. Download it to `data/process2/` and build the metadata:

```bash
cd scriptsUpdates
python process2_preprocess.py                       # Cookie Theft task; healthy vs MCI+dementia
python process2_preprocess.py --labels dementia     # healthy vs dementia only
python process2_preprocess.py --tasks CTD PFT SFT   # include the fluency tasks
```

With `USE_OFFICIAL_SPLIT = True` (default), training uses the dataset's own train/test split.

### DementiaBank Pitt

The [DementiaBank](https://talkbank.org/dementia/) Pitt corpus (Cookie Theft picture description) requires DementiaBank membership. Expected layout (audio may be `.wav` or `.mp3`; transcripts share the audio file's name):

```
data/pitt/control/audio/*.wav         data/pitt/control/transcripts/*.cha
data/pitt/dementia/audio/*.wav        data/pitt/dementia/transcripts/*.cha
```

---

## Usage

Run the scripts from `scriptsUpdates/` (paths in `enhanced_config.py` are relative to it):

```bash
cd scriptsUpdates
python enhanced_preprocess.py   # Pitt; or process2_preprocess.py for PROCESS-2
python enhanced_train.py        # writes ../results and ../models
python enhanced_results.py      # summary report and plots in ../results
```

### Evaluation protocol

- PROCESS-2: the official train/test split, with a validation set held out from the training participants.
- Otherwise: 10-fold cross-validation; within each fold the remaining data is split 8 : 1 into train and validation.
- Splits are **stratified** by label and **grouped by speaker**, so recordings from the same participant never appear in both training and test data.
- Prosodic/paralinguistic features are standardized using statistics from the training split only.
- The checkpoint with the best validation F1 is evaluated on the held-out test fold.

### Training details

AdamW (1e-5 for the pretrained backbones, 1e-4 for the new layers), cosine schedule with warmup, class-weighted cross-entropy with label smoothing, gradient clipping and early stopping (patience 5).

---

## Results

**PROCESS-2, official train/test split.** Healthy controls (0) vs. cognitively impaired, MCI or dementia (1); Cookie Theft recordings only; 400 participants (200 / 200). Participants: 284 train, 36 validation, 80 test. Single run (seed 42) on a Kaggle T4. The checkpoint with the best validation F1 (epoch 11; early stopping at epoch 16) was evaluated once on the test set.

| Metric | Test set (80 participants) |
|---|---|
| Accuracy | 81.3% |
| F1 | 0.815 |
| Precision | 0.805 |
| Recall | 0.825 |
| ROC-AUC | 0.919 |

Confusion matrix: 32 true negatives, 8 false positives, 7 false negatives, 33 true positives.

With 80 test participants the 95% confidence interval for accuracy is roughly 71–88%, so differences of a few points between runs are not meaningful. Training accuracy reached ~97% while validation peaked near 80%, so the model overfits after about 10 epochs; early stopping limits this.

---

## Troubleshooting

- **CUDA out of memory** — lower `BATCH_SIZE` or `MAX_AUDIO_SECONDS` in `enhanced_config.py`.
- **"Error loading" audio files** — the paths in `metadata.csv` don't match your data location; check `DATA_DIR` and re-run `enhanced_preprocess.py`.
- **Changed preprocessing** — delete `data/handcrafted_features.npz` so the cached features are recomputed.
