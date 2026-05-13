# 🧠 EEG Mind Reader

**Decoding Imagined Movements from Brain Signals Using Deep Learning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-blueviolet.svg)](https://claude.ai)

---

> **What if a computer could read your thoughts from brain signals?**
>
> This project decodes **motor imagery** from EEG — predicting whether a person is imagining moving their left hand, right hand, feet, or tongue based purely on their brainwaves. This is the foundational technology behind Brain-Computer Interfaces (BCI) used in prosthetics, assistive devices for paralyzed patients, and next-generation interfaces like Neuralink.

---

## 🎬 Live Demo

<div align="center">

![BCI Dashboard Demo](reports/figures/bci_demo.gif)

*Cyberpunk-style neural interface decoding brain signals in real-time*

</div>

### 🎮 Run the Interactive Demo

```bash
# Activate virtual environment first
cd eeg-mind-reader
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Mac/Linux

# Launch the BCI Dashboard
streamlit run app/bci_dashboard.py
```

The dashboard opens at `http://localhost:8501` with two modes:

| Mode | What It Does |
|------|--------------|
| **🎯 Mind Cursor** | A game where decoded brain signals control a cursor — watch it move left or right based on imagined movements! |
| **🧠 Brain Theater** | Full cinematic visualization with streaming EEG, 3D brain activity, and real-time classification |

---

## 🧠 What You're Seeing (For Non-ML Readers)

### The Mind Cursor Game

1. **The Data**: We load a 2-second recording of someone's brain waves (22 electrodes on their scalp)
2. **The Task**: The person was asked to *imagine* moving their left or right hand (without actually moving)
3. **The AI**: Our neural network analyzes the brain patterns and predicts which hand they imagined
4. **The Cursor**: Moves left or right based on the AI's confidence — if it's 70% sure "right hand", the cursor drifts right
5. **The Score**: Tracks how often the AI correctly decodes the person's imagined movement

### The Brain Theater

- **Top Panel**: Raw EEG signals streaming from 8 key electrodes over the motor cortex
- **3D Brain**: Electrodes light up based on activity — brighter = more active
- **Probability Bars**: The AI's confidence for each of the 4 possible imagined movements
- **Big Prediction**: The AI's current best guess, updating multiple times per second

**The magic**: The person isn't moving at all — we're reading their *thoughts* about movement directly from their brain!

---

## 🎯 The Problem

**Motor imagery** is the mental rehearsal of movement without physical execution. When you imagine moving your left hand, your brain generates distinct electrical patterns over the motor cortex — patterns that can be detected with EEG electrodes on the scalp.

The challenge: these signals are **incredibly noisy** (microvolts buried in artifacts), vary **dramatically between individuals**, and require sophisticated algorithms to decode reliably.

### Why This Matters

| Application | Impact |
|-------------|--------|
| 🦾 **Prosthetics** | Control robotic limbs with thought alone |
| 💬 **Communication** | Help locked-in syndrome patients express themselves |
| 🏥 **Neurorehabilitation** | Accelerate stroke recovery through neurofeedback |
| 🎮 **Gaming/VR** | Hands-free immersive control |
| 🧬 **Neuralink** | Foundation for next-gen brain-computer interfaces |

---

## 🔬 The Approach

We implement and compare **three deep learning architectures** on the benchmark BCI Competition IV Dataset 2a:

### Models

| Model | Architecture | Parameters | Strength |
|-------|--------------|------------|----------|
| **EEGNet** | Depthwise Separable CNN | ~2,600 | Compact, efficient, interpretable |
| **CNN-LSTM** | CNN + Bidirectional LSTM + Attention | ~200,000 | Captures long-range temporal dynamics |
| **Transformer** | Patch Embedding + Self-Attention | ~150,000 | Global context, attention visualization |

### The Real Challenge: Cross-Subject Generalization

Within-subject classification (train and test on same person) is relatively easy. The **real** BCI challenge is:

> *Can a model trained on some people's brains work on a completely new person?*

We evaluate this using **Leave-One-Subject-Out (LOSO)** cross-validation — training on 8 subjects and testing on the held-out 9th, repeated for all 9 subjects.

---

## 📊 Results

### Within-Subject Classification (Subject 1)

| Model | Accuracy | Balanced Acc | F1 (macro) | Training Time |
|-------|----------|--------------|------------|---------------|
| EEGNet | `0.XX` | `0.XX` | `0.XX` | `XX.Xs` |
| CNN-LSTM | `0.XX` | `0.XX` | `0.XX` | `XX.Xs` |
| Transformer | `0.XX` | `0.XX` | `0.XX` | `XX.Xs` |

### Cross-Subject (LOSO) Classification

| Model | Mean Accuracy | Std | Min | Max |
|-------|---------------|-----|-----|-----|
| EEGNet | `0.XX ± 0.XX` | `0.XX` | `0.XX` | `0.XX` |
| CNN-LSTM | `0.XX ± 0.XX` | `0.XX` | `0.XX` | `0.XX` |
| Transformer | `0.XX ± 0.XX` | `0.XX` | `0.XX` | `0.XX` |

*Fill in after running the notebook*

---

## 📈 Key Visualizations

<table>
<tr>
<td width="50%">

### Confusion Matrix
![Confusion Matrix](reports/figures/eegnet_confusion_matrix.png)
*EEGNet classification performance across 4 motor imagery classes*

</td>
<td width="50%">

### t-SNE Feature Space
![t-SNE](reports/figures/tsne_eegnet.png)
*Learned representations show class separation*

</td>
</tr>
<tr>
<td width="50%">

### Channel Importance
![Channel Importance](reports/figures/channel_importance_eegnet.png)
*Model focuses on motor cortex (C3, C4, Cz) — neurologically plausible!*

</td>
<td width="50%">

### Within vs Cross-Subject
![Comparison](reports/figures/within_vs_loso.png)
*The generalization gap reveals inter-subject variability*

</td>
</tr>
</table>

---

## 🗂️ Project Structure

```
eeg-mind-reader/
├── 📓 notebooks/
│   └── eeg_classification.ipynb    # Main analysis notebook (start here!)
├── 📦 src/
│   ├── config.py                   # All hyperparameters & paths
│   ├── data_loader.py              # MOABB dataset loading
│   ├── preprocessing.py            # Filtering, epoching, normalization
│   ├── models/
│   │   ├── eegnet.py               # EEGNet architecture
│   │   ├── cnn_lstm.py             # CNN-LSTM hybrid
│   │   └── transformer.py          # EEG Transformer
│   ├── training.py                 # Training loop with early stopping
│   ├── evaluation.py               # Metrics & LOSO evaluation
│   └── visualization.py            # All plotting functions
├── 🎮 app/
│   ├── bci_dashboard.py            # 🔥 Cyberpunk BCI demo (Mind Cursor + Brain Theater)
│   ├── streamlit_app.py            # Simple live decoder
│   └── record_demo.py              # Auto-play for GIF recording
├── 🧪 tests/
│   └── test_pipeline.py            # Smoke tests
├── 📊 data/
│   ├── raw/                        # Auto-downloaded by MOABB
│   └── processed/                  # Cached preprocessed data
├── 🏆 models/                      # Saved checkpoints
├── 📈 reports/figures/             # Generated plots
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/eeg-mind-reader.git
cd eeg-mind-reader

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook (generates all results)

```bash
jupyter notebook notebooks/eeg_classification.ipynb
```

The notebook will:
- Auto-download the dataset via MOABB (first run only)
- Train all three models on Subject 1
- Generate all visualizations
- Optionally run full LOSO evaluation

### 3. Launch the BCI Dashboard

```bash
streamlit run app/bci_dashboard.py
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch 2.0+ |
| **EEG Processing** | MNE-Python, MOABB |
| **Data Science** | NumPy, Pandas, scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Demo App** | Streamlit |
| **Testing** | pytest |

---

## 📚 Dataset

**BCI Competition IV Dataset 2a**

- **Subjects:** 9 healthy participants
- **Channels:** 22 EEG electrodes (10-20 system)
- **Classes:** 4 (left hand, right hand, feet, tongue)
- **Sampling Rate:** 250 Hz
- **Trials:** ~288 per subject

The dataset is automatically downloaded via [MOABB](https://github.com/NeuroTechX/moabb) on first run.

---

## 🧪 Reproducibility

All experiments are fully reproducible:

```python
# Seeds are set in config.py
SEED = 42
set_seed(SEED)  # Sets random, numpy, torch, cuda seeds
```

Model checkpoints are saved to `models/` and can be reloaded for inference.

---

## 📖 References

1. **Lawhern, V. J., et al.** (2018). *EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces.* Journal of Neural Engineering, 15(5). [Paper](https://doi.org/10.1088/1741-2552/aace8c)

2. **Tangermann, M., et al.** (2012). *Review of the BCI Competition IV.* Frontiers in Neuroscience. [Paper](https://doi.org/10.3389/fnins.2012.00055)

3. **Jayaram, V., & Barachant, A.** (2018). *MOABB: Trustworthy algorithm benchmarking for BCIs.* Journal of Neural Engineering. [Paper](https://doi.org/10.1088/1741-2552/aab2ab)

4. **Vaswani, A., et al.** (2017). *Attention Is All You Need.* NeurIPS. [Paper](https://arxiv.org/abs/1706.03762)

---

## 🔮 Future Work

- [ ] **Subject-Adaptive Transfer Learning** — Fine-tune with minimal calibration data
- [ ] **Domain Adaptation** — Align feature distributions across subjects
- [ ] **Data Augmentation** — Generate synthetic EEG for larger training sets
- [ ] **Real-Time Inference** — Optimize for <100ms latency
- [ ] **Multimodal Fusion** — Combine EEG with EMG or eye tracking

---

## 👤 Author

**Sanober**

- Email: maaz@startupfuel.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [Your GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🧠 and ☕**

*If you found this project interesting, please consider giving it a ⭐!*

</div>
