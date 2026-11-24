"""
Task 1: Dataset Loading, Verification, and Exploration for RadioML RML2016.10a.

Usage:
    python -m src.data_loader --path data/raw/RML2016.10a_dict.pkl
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_rml2016a(filepath: str) -> Dict[Tuple[str, int], np.ndarray]:
    """Load the RML2016.10a pickle file.

    Note: RML2016.10a was originally serialized with Python 2, so `encoding='latin1'`
    is required when loading in Python 3.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'.\n"
            "Download RML2016.10a from https://www.deepsig.ai/datasets "
            "and place 'RML2016.10a_dict.pkl' in 'data/raw/'."
        )

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def summarize_dataset(filepath: str, data: Dict[Tuple[str, int], np.ndarray]) -> dict:
    """Inspect and return key verification metadata for RML2016.10a."""
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)

    modulations = sorted({mod for mod, _ in data.keys()})
    snrs = sorted({snr for _, snr in data.keys()})

    sample_key = next(iter(data.keys()))
    sample_arr = data[sample_key]
    per_pair_counts = {k: v.shape[0] for k, v in data.items()}
    total_examples = sum(per_pair_counts.values())

    unique_counts = set(per_pair_counts.values())
    is_balanced = len(unique_counts) == 1

    return {
        "file_name": Path(filepath).name,
        "format": Path(filepath).suffix,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "num_modulations": len(modulations),
        "modulations": modulations,
        "num_snrs": len(snrs),
        "snrs": snrs,
        "total_examples": total_examples,
        "examples_per_mod_snr": next(iter(unique_counts)) if is_balanced else "varies",
        "is_balanced": is_balanced,
        "observation_shape": sample_arr.shape[1:],
        "dtype": str(sample_arr.dtype),
    }


def print_summary(summary: dict) -> None:
    """Pretty-print the dataset verification summary."""
    print("=" * 60)
    print("RML2016.10a Dataset Verification Summary")
    print("=" * 60)
    print(f"File name          : {summary['file_name']}")
    print(f"Format             : {summary['format']}")
    print(f"File size          : {summary['file_size_mb']} MB ({summary['file_size_bytes']:,} bytes)")
    print(f"Modulation classes : {summary['num_modulations']} -> {summary['modulations']}")
    print(f"SNR levels (dB)    : {summary['num_snrs']} -> {summary['snrs']}")
    print(f"Total examples     : {summary['total_examples']:,}")
    print(f"Examples/(mod,SNR) : {summary['examples_per_mod_snr']} (balanced={summary['is_balanced']})")
    print(f"Observation shape  : {summary['observation_shape']} (2=I/Q, 128=time samples)")
    print(f"Data type          : {summary['dtype']}")
    print("=" * 60)


def build_arrays(
    data: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Convert raw dictionary into (X, y, snrs, class_names) arrays.

    Returns:
        X: shape (N, 2, 128), float32
        y: shape (N,), int64 class indices (0..10)
        snr_arr: shape (N,), int64 SNR values in dB
        modulations: sorted list of modulation class names matching indices 0..10
    """
    modulations = sorted({mod for mod, _ in data.keys()})
    mod_to_idx = {mod: idx for idx, mod in enumerate(modulations)}

    X_list, y_list, snr_list = [], [], []
    for (mod, snr), signals in sorted(data.items()):
        n = signals.shape[0]
        X_list.append(signals)
        y_list.append(np.full(n, mod_to_idx[mod], dtype=np.int64))
        snr_list.append(np.full(n, snr, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    snr_arr = np.concatenate(snr_list)
    return X, y, snr_arr, modulations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and explore RML2016.10a dataset.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw/RML2016.10a_dict.pkl",
        help="Path to RML2016.10a_dict.pkl",
    )
    args = parser.parse_args()

    raw_data = load_rml2016a(args.path)
    info = summarize_dataset(args.path, raw_data)
    print_summary(info)
