# Automatic Modulation Classification (AMC) Using Deep Learning

**Course:** Wireless Communication Networks and Systems (WCNS) — ENSIA, 4th Year  
**Dataset:** RadioML `RML2016.10a` (11 modulations, 20 SNR levels from −20 dB to +18 dB, 2×128 I/Q frames)

## Repository Structure

```text
project/
├── configs/
│   └── config.yaml         # Seeds, paths, split ratios, visualization settings
├── data/
│   └── raw/                # Place RML2016.10a_dict.pkl here (git-ignored)
├── outputs/
│   ├── figures/            # Generated plots (time, constellation, FFT, CM, etc.)
│   └── models/             # Saved model weights
├── src/
│   ├── data_loader.py      # Task 1: Load & verify RML2016.10a
│   └── visualization.py    # Task 2: Time-domain, I/Q constellation & FFT plots
├── References/             # Provided reference papers
├── requirements.txt
└── README.md
```

## Quick Start

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Download the dataset:**
   Download `RML2016.10a` from [DeepSig Datasets](https://www.deepsig.ai/datasets) and place `RML2016.10a_dict.pkl` inside `data/raw/`.
3. **Run Task 1 (Dataset Verification & Exploration):**
   ```bash
   python -m src.data_loader
   ```
4. **Run Task 2 (Signal Visualizations):**
   ```bash
   python -m src.visualization
   ```
