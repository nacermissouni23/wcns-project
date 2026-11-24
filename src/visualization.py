"""
Task 2: Signal Visualization and Analysis for RadioML RML2016.10a.

Generates:
  A. Time-domain plots (I, Q, and I+Q together across SNRs)
  B. I/Q constellation diagrams across SNRs
  C. Frequency-domain magnitude spectra (FFT with Hann windowing)

Usage:
    python -m src.visualization --path data/raw/RML2016.10a_dict.pkl
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_rml2016a

DEFAULT_MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "AM-DSB", "GFSK"]
DEFAULT_SNRS = [-10, 0, 10, 18]


def plot_time_domain(
    data: Dict[Tuple[str, int], np.ndarray],
    mod: str,
    snrs: List[int],
    sample_idx: int = 0,
    output_dir: Path | None = None,
) -> None:
    """Plot I, Q, and combined I/Q time-domain waveforms across SNR levels."""
    fig, axes = plt.subplots(len(snrs), 1, figsize=(10, 2.5 * len(snrs)), sharex=True)
    if len(snrs) == 1:
        axes = [axes]

    t = np.arange(128)
    for ax, snr in zip(axes, snrs):
        iq = data[(mod, snr)][sample_idx]
        ax.plot(t, iq[0], label="I (In-phase)", linewidth=1.2)
        ax.plot(t, iq[1], label="Q (Quadrature)", linewidth=1.2, alpha=0.85)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{mod} — Time Domain (SNR = {snr:+d} dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Sample Index (n)")
    fig.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"time_{mod}.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_constellation_grid(
    data: Dict[Tuple[str, int], np.ndarray],
    modulations: List[str],
    snrs: List[int],
    n_frames: int = 5,
    output_dir: Path | None = None,
) -> None:
    """Plot I/Q constellation diagrams for selected modulations across SNRs."""
    fig, axes = plt.subplots(
        len(modulations),
        len(snrs),
        figsize=(3.2 * len(snrs), 2.8 * len(modulations)),
    )

    for row, mod in enumerate(modulations):
        for col, snr in enumerate(snrs):
            ax = axes[row, col]
            frames = data[(mod, snr)][:n_frames]
            i_vals = frames[:, 0, :].ravel()
            q_vals = frames[:, 1, :].ravel()
            ax.scatter(i_vals, q_vals, s=4, alpha=0.5)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(f"SNR = {snr:+d} dB", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{mod}\nQ", fontsize=9, fontweight="bold")
            if row == len(modulations) - 1:
                ax.set_xlabel("I")

    fig.suptitle("I/Q Constellation Diagrams Across SNR Levels", fontsize=13, y=1.01)
    fig.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "constellations_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_fft_grid(
    data: Dict[Tuple[str, int], np.ndarray],
    modulations: List[str],
    snrs: List[int],
    sample_idx: int = 0,
    fft_len: int = 128,
    output_dir: Path | None = None,
) -> None:
    """Plot centered FFT magnitude spectrum (dB, Hann window) across modulations and SNRs."""
    window = np.hanning(fft_len)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len))

    fig, axes = plt.subplots(
        len(modulations),
        len(snrs),
        figsize=(3.4 * len(snrs), 2.4 * len(modulations)),
        sharex=True,
    )

    for row, mod in enumerate(modulations):
        for col, snr in enumerate(snrs):
            ax = axes[row, col]
            iq = data[(mod, snr)][sample_idx]
            complex_sig = (iq[0] + 1j * iq[1]) * window
            spectrum = np.fft.fftshift(np.fft.fft(complex_sig, n=fft_len))
            mag_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
            ax.plot(freqs, mag_db, linewidth=1.0)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"SNR = {snr:+d} dB", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{mod}\nMag (dB)", fontsize=9, fontweight="bold")
            if row == len(modulations) - 1:
                ax.set_xlabel("Normalized Freq")

    fig.suptitle(f"FFT Magnitude Spectrum (N={fft_len}, Hann Window)", fontsize=13, y=1.01)
    fig.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "fft_spectra_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Task 2 signal visualizations.")
    parser.add_argument("--path", type=str, default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--out", type=str, default="outputs/figures")
    args = parser.parse_args()

    raw_data = load_rml2016a(args.path)
    out_dir = Path(args.out)

    for m in DEFAULT_MODULATIONS:
        plot_time_domain(raw_data, m, DEFAULT_SNRS, output_dir=out_dir)

    plot_constellation_grid(raw_data, DEFAULT_MODULATIONS, DEFAULT_SNRS, output_dir=out_dir)
    plot_fft_grid(raw_data, DEFAULT_MODULATIONS, DEFAULT_SNRS, output_dir=out_dir)
    print(f"Saved all Task 2 figures to {out_dir.resolve()}")
