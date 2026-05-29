from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, raw_dir, segment_length=1000, max_files=100):
        self.raw_dir = Path(raw_dir)
        self.segment_length = segment_length
        self.max_files = max_files
        self.segments = self._load_segments()

        if not self.segments:
            raise ValueError(f"No valid ECG segments found in {self.raw_dir}")

    def _load_segments(self):
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if self.max_files is not None:
            csv_files = csv_files[: self.max_files]

        segments = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, header=None)
            except Exception as exc:
                print(f"Skip {file_path}: {exc}")
                continue

            if df.shape[1] < 2:
                print(f"Skip {file_path}: expected at least 2 columns, got {df.shape[1]}")
                continue

            signal = df.iloc[:, 1].dropna().to_numpy(dtype=np.float32)
            if signal.shape[0] < self.segment_length:
                continue

            signal_min = float(signal.min())
            signal_max = float(signal.max())
            if signal_max > signal_min:
                signal = 2.0 * (signal - signal_min) / (signal_max - signal_min) - 1.0
            else:
                signal = np.zeros_like(signal, dtype=np.float32)

            num_segments = signal.shape[0] // self.segment_length
            for idx in range(num_segments):
                start = idx * self.segment_length
                end = start + self.segment_length
                segments.append(signal[start:end].astype(np.float32))

        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        clean_signal = torch.from_numpy(self.segments[index]).unsqueeze(0)
        return clean_signal
