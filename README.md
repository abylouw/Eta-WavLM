# Eta-WavLM — Non-Official Implementation

*A linear decomposition of SSL speech representations into speaker and content components*
**arXiv:2505.19273v1**

This repository contains a **non-official research implementation** of the methodology introduced in:

> **η-WavLM: Linear Decomposition of Speaker and Content Components in Self-Supervised Speech Representations**

The implementation aims to separate SSL representations into:

* a **speaker-predictable linear component**, and
* a **speaker-independent residual**, denoted **η**.

The system closely follows the analytic framework in the paper, using **streamed least-squares accumulation** and a **closed-form Cholesky solver**, without gradient-based optimisation.

---

# ✨ Method Overview

Given:

* SSL frame features (s \in \mathbb{R}^Q),
* speaker embeddings (d \in \mathbb{R}^P),

the representation is decomposed as:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=s%20%3D%20f(d)%20%2B%20%5Ceta%20%3D%20A%5E%5Ctop%20d%20%2B%20b%20%2B%20%5Ceta">
</div>

The augmented speaker matrix is:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Ctilde{D}%20%3D%20%5BD%5C%20%5C%201%5D">
</div>

The mapping ((A,b)) is obtained by solving:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=(%5Ctilde{D}%5E%5Ctop%20%5Ctilde{D})%20A%20%3D%20%5Ctilde{D}%5E%5Ctop%20S">
</div>

This repository implements the process with:

* **WavLM-Large** for SSL features
* **ECAPA-TDNN** for speaker embeddings
* **PCA** reduced to 128 dimensions
* **Double-precision accumulation** and **regularised Cholesky solve**

---

# 📍 Two-Stage Training Pipeline

## **Stage 1 — Speaker Embedding Extraction and PCA**

For each utterance:

1. Extract ECAPA-TDNN 192-dim embeddings
2. Fit PCA across the dataset
3. Reduce embeddings to **128 dimensions**
4. Save

   * PCA-reduced embeddings
   * PCA model
   * metadata

This stage is executed once per corpus.

---

## **Stage 2 — Linear Decomposition (Closed-Form Training)**

For each utterance:

1. Extract WavLM SSL features
2. Sample **L = 100 random frames**
3. Load the PCA-reduced 128-dim speaker embedding
4. Duplicate the speaker vector across all sampled frames
5. Accumulate the normal-equation terms:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=G%20%2B%3D%20%5Ctilde{D}%5E%5Ctop%20%5Ctilde{D}%2C%20%5Cqquad%20H%20%2B%3D%20%5Ctilde{D}%5E%5Ctop%20S">
</div>

After all batches:

* Solve once using Cholesky
* Obtain matrices (A) and (b)
* No gradients or back-propagation are used

---

# 📍 Validation

Validation computes η-representations and evaluates speaker leakage:

1. Extract WavLM features
2. Extract ECAPA-TDNN → PCA speaker embedding
3. Compute projection:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=f(d)%20%3D%20A%5E%5Ctop%20d%20%2B%20b">
</div>

4. Compute η residual:

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Ceta%20%3D%20s%20-%20f(d)">
</div>

5. Train SVM classifiers on:

   * SSL features
   * η features

6. Measure reduction in speaker classification accuracy.

Example (LibriSpeech ~1000h model):

```
val_ssl_speaker_accuracy      = 80.5%
val_eta_speaker_accuracy      = 77.4%
speaker_accuracy_reduction    = 3.1 percentage points
```

---

# 📥 Dataset Download Utility

Datasets may be downloaded automatically:

### List available datasets

```bash
python scripts/download_datasets.py --list
```

### LibriTTS

```bash
python scripts/download_datasets.py --dataset libri-tts --splits dev-clean test-clean
```

### LibriSpeech

```bash
python scripts/download_datasets.py --dataset libri-speech --splits train-clean-100 dev-clean
```

### VCTK

```bash
python scripts/download_datasets.py --dataset vctk --splits vctk-corpus-0.92
```

### Verify checksums

```bash
python scripts/download_datasets.py --verify libri-tts
```

---

# 🖥️ Model Server (LitServe)

The repository includes a **LitServe API server** for real-time η-vector computation.
It works with the released **LibriSpeech ~1000-hour model** included in this repository.

### Start the server

```bash
python scripts/serve.py \
  --model-path outputs/eta_wavlm_full_training/eta_wavlm_full_training_model.pt \
  --config-path outputs/eta_wavlm_full_training/eta_wavlm_full_training_config.yaml
```

The server automatically:

* loads the model
* loads the PCA speaker model
* runs WavLM and ECAPA-TDNN
* resamples audio to **16kHz**
* returns SSL, speaker, and η-features

---

# 📡 Client Program (`client.py`)

A reference client is provided for testing the API with WAV files.

### Usage

```bash
python scripts/client.py -h
```

```
usage: client.py [-h] [--api-url API_URL] [--save-results SAVE_RESULTS]
                 [--validate-only]
                 wav_path
```

### Example

```bash
python scripts/client.py phase4_learner_041_male_8_1862.wav
```

### Example Output (Truncated)

```
WAV file validation passed
Sending request to API...
Decomposition successful!

======================================================================
DECOMPOSITION ANALYSIS
======================================================================
SSL Features:
  Shape: (100, 1024)
  Mean: 0.302616
  Std: 8.163890

Speaker Embedding:
  Shape: (100, 128)

Eta Features:
  Shape: (100, 1024)
  Mean: 0.049078
  Std: 7.062877

Feature Magnitudes:
  SSL norm: 2614.23
  Eta norm: 2260.17
  Speaker/SSL ratio: 0.9588
  Eta/SSL ratio: 0.8645
```

---

# 📦 Repository Structure

```
.
├── cache/
├── checkpoints/
├── configs/
│   ├── data/
│   ├── model/
│   └── train.yaml
├── data/
├── logs/
├── outputs/
├── scripts/
│   ├── download_datasets.py
│   ├── serve.py
│   ├── client.py
│   └── train.py
└── src/
    ├── data/
    ├── features/
    ├── models/
    └── utils/
```

---

# ⚙️ Training Usage

### Standard Training

```bash
python scripts/train.py --config configs/train.yaml
```

### Validation Only

```bash
python scripts/train.py \
    --config configs/train.yaml \
    --validate-only \
    --model-path path/to/model.pt
```

### Custom Experiment Location

```bash
python scripts/train.py \
    --config configs/train.yaml \
    --experiment-name my_experiment \
    --output-dir outputs/custom
```

---

# 📦 Requirements

```
lightning==2.5.6
lightning-utilities==0.15.2
pytorch-lightning==2.5.6
soundfile==0.13.1
speechbrain @ git+https://github.com/speechbrain/speechbrain.git@b7afd0350a9f6acaf8aa6dcd7dcffec25fa82e30
torchmetrics==1.8.2
tqdm==4.67.1
transformers==4.57.1
scikit-learn==1.7.2
torch==2.9.1
torchaudio==2.9.1 
tensorboard==2.20.0
rich==10.2.2
litserve==0.2.16
```

---

# 🛠️ Installation

## **CPU-only**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cpu.txt
```

## **CUDA 12.6 (NVIDIA GPU)**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cuda.txt
```

---

# 🧪 Current Status

This repository implements the full η-WavLM analytic decomposition pipeline.
However:

* Speaker leakage reduction remains lower than reported in the original paper
* Dataset scale, embedding normalisation, and PCA interactions may contribute
* Additional experiments are ongoing

Community contributions, issue reports, and replication attempts are welcome.

---

# 📚 Citation

```
@article{eta-wavlm,
  title={η-WavLM: Linear Decomposition of Speaker and Content Components in SSL Speech Representations},
  journal={arXiv preprint arXiv:2505.19273},
  year={2025}
}
```

This repository is not affiliated with the authors of the original publication.

---

