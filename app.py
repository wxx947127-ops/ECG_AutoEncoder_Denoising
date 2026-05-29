from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ECGAutoEncoder
from noise import add_noise


SEGMENT_LENGTH = 1000
MODEL_PATH = PROJECT_ROOT / "results" / "best_model.pth"
NOISE_TYPE_MAP = {
    "Mixed Noise": "mixed",
    "Gaussian Noise": "gaussian",
    "Powerline Interference": "line",
    "Baseline Wander": "baseline",
}


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGAutoEncoder().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def load_ecg_signal(uploaded_file):
    df = pd.read_csv(uploaded_file, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns. Column 2 is used as ECG signal.")

    signal = df.iloc[:, 1].dropna().to_numpy(dtype=np.float32)
    if len(signal) < SEGMENT_LENGTH:
        raise ValueError(f"ECG signal length must be at least {SEGMENT_LENGTH}.")

    signal = signal[:SEGMENT_LENGTH]
    signal_min = float(signal.min())
    signal_max = float(signal.max())
    if signal_max > signal_min:
        signal = 2.0 * (signal - signal_min) / (signal_max - signal_min) - 1.0
    else:
        signal = np.zeros_like(signal, dtype=np.float32)
    return signal.astype(np.float32)


def run_autoencoder(model, device, clean_signal, noise_type):
    clean_tensor = torch.from_numpy(clean_signal).float().view(1, 1, SEGMENT_LENGTH).to(device)
    torch.manual_seed(42)
    noisy_tensor = add_noise(clean_tensor, noise_type)

    with torch.no_grad():
        predicted_noise = model(noisy_tensor)
        denoised_tensor = torch.clamp(noisy_tensor - predicted_noise, -1.0, 1.0)

    noisy_signal = noisy_tensor.squeeze().detach().cpu().numpy()
    denoised_signal = denoised_tensor.squeeze().detach().cpu().numpy()
    return noisy_signal, denoised_signal


def calculate_snr(clean_signal, compared_signal):
    noise = clean_signal - compared_signal
    signal_power = float(np.mean(clean_signal**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 1e-12:
        return 99.0
    return 10.0 * np.log10(signal_power / noise_power)


def calculate_pearson(clean_signal, compared_signal):
    clean_std = float(np.std(clean_signal))
    compared_std = float(np.std(compared_signal))
    if clean_std <= 1e-12 or compared_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(clean_signal, compared_signal)[0, 1])


def calculate_metrics(clean_signal, noisy_signal, denoised_signal):
    snr_before = calculate_snr(clean_signal, noisy_signal)
    snr_after = calculate_snr(clean_signal, denoised_signal)
    return {
        "MSE": float(np.mean((denoised_signal - clean_signal) ** 2)),
        "MAE": float(np.mean(np.abs(denoised_signal - clean_signal))),
        "Pearson": calculate_pearson(clean_signal, denoised_signal),
        "SNR_before": snr_before,
        "SNR_after": snr_after,
        "SNR_improvement": snr_after - snr_before,
    }


def detect_r_peaks(signal):
    prominence = max(0.15, float(np.std(signal)) * 0.8)
    peaks, _ = find_peaks(signal, distance=150, prominence=prominence)
    if len(peaks) == 0:
        height = float(np.percentile(signal, 90))
        peaks, _ = find_peaks(signal, distance=150, height=height)
    return peaks.astype(np.int64)


def draw_peak_windows(axis, peaks, start_offset, end_offset, color, label, text_y):
    for peak in peaks:
        start = max(0, int(peak + start_offset))
        end = min(SEGMENT_LENGTH - 1, int(peak + end_offset))
        if start >= end:
            continue
        axis.axvspan(start, end, color=color, alpha=0.16)
        axis.text(
            (start + end) / 2,
            text_y,
            label,
            color=color,
            fontsize=9,
            ha="center",
            va="center",
        )


def draw_annotations(axis, signal, r_peaks, show_p, show_qrs, show_t, show_r):
    if len(r_peaks) == 0:
        return

    if show_p:
        draw_peak_windows(axis, r_peaks, -200, -120, "green", "P", 0.58)
    if show_qrs:
        draw_peak_windows(axis, r_peaks, -20, 20, "red", "QRS", 0.84)
    if show_t:
        draw_peak_windows(axis, r_peaks, 100, 250, "blue", "T", 0.68)
    if show_r:
        valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
        axis.scatter(valid_peaks, signal[valid_peaks], color="red", s=28, marker="o", zorder=5)
        for peak in valid_peaks:
            text_y = min(float(signal[peak]) + 0.1, 1.02)
            axis.text(peak, text_y, "R", color="red", fontsize=9, ha="center")


def plot_signals(clean_signal, noisy_signal, denoised_signal, feature_options):
    r_peaks = detect_r_peaks(clean_signal)
    signals = [
        ("Original Signal", "Original Signal", "green", clean_signal),
        ("Noisy Signal", "Noisy Signal", "red", noisy_signal),
        ("Denoised Signal", "Denoised Signal", "blue", denoised_signal),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=120)
    x = np.arange(SEGMENT_LENGTH)

    for axis, (title, label, color, signal) in zip(axes, signals):
        axis.plot(x, signal, color=color, linewidth=1.2, label=label)
        draw_annotations(axis, signal, r_peaks, **feature_options)
        axis.set_title(title)
        axis.set_xlabel("Sample")
        axis.set_ylabel("ECG Signal")
        axis.set_xlim(0, SEGMENT_LENGTH - 1)
        axis.set_ylim(-1.08, 1.08)
        axis.grid(True, alpha=0.5)
        axis.legend(loc="lower left")

    fig.tight_layout()
    return fig


def make_signal_csv(clean_signal, noisy_signal, denoised_signal):
    signal_df = pd.DataFrame(
        {
            "clean_signal": clean_signal,
            "noisy_signal": noisy_signal,
            "denoised_signal": denoised_signal,
        }
    )
    return signal_df.to_csv(index=False).encode("utf-8-sig")


def make_metrics_csv(metrics):
    metrics_df = pd.DataFrame(
        [
            {
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "Pearson": metrics["Pearson"],
                "SNR_before": metrics["SNR_before"],
                "SNR_after": metrics["SNR_after"],
                "SNR_improvement": metrics["SNR_improvement"],
            }
        ]
    )
    return metrics_df.to_csv(index=False).encode("utf-8-sig")


def make_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def show_download_area(clean_signal, noisy_signal, denoised_signal, metrics, fig):
    st.subheader("Download Results")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        label="Download denoised_signal.csv",
        data=make_signal_csv(clean_signal, noisy_signal, denoised_signal),
        file_name="denoised_signal.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        label="Download metrics.csv",
        data=make_metrics_csv(metrics),
        file_name="metrics.csv",
        mime="text/csv",
    )
    download_cols[2].download_button(
        label="Download result.png",
        data=make_png(fig),
        file_name="result.png",
        mime="image/png",
    )


def main():
    st.set_page_config(page_title="ECG Denoising System", layout="wide")
    st.title("ECG Denoising System")

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        noise_label = st.selectbox("Noise Type", list(NOISE_TYPE_MAP.keys()), index=0)

        st.header("Waveform Features")
        show_p = st.checkbox("Show P Wave", value=False)
        show_qrs = st.checkbox("Show QRS Complex", value=False)
        show_t = st.checkbox("Show T Wave", value=False)
        show_r = st.checkbox("Show R Peak", value=False)

    if uploaded_file is None:
        st.info("Please upload a headerless 12-column ECG CSV file. Column 2 is used by default.")
        return

    try:
        model, device = load_model()
        clean_signal = load_ecg_signal(uploaded_file)
        noisy_signal, denoised_signal = run_autoencoder(
            model, device, clean_signal, NOISE_TYPE_MAP[noise_label]
        )
        metrics = calculate_metrics(clean_signal, noisy_signal, denoised_signal)

        metric_cols = st.columns(4)
        metric_cols[0].metric("MSE", f"{metrics['MSE']:.6f}")
        metric_cols[1].metric("MAE", f"{metrics['MAE']:.6f}")
        metric_cols[2].metric("Pearson", f"{metrics['Pearson']:.4f}")
        metric_cols[3].metric("SNR Improvement", f"{metrics['SNR_improvement']:.2f} dB")

        feature_options = {
            "show_p": show_p,
            "show_qrs": show_qrs,
            "show_t": show_t,
            "show_r": show_r,
        }
        fig = plot_signals(clean_signal, noisy_signal, denoised_signal, feature_options)
        st.pyplot(fig, clear_figure=False)
        show_download_area(clean_signal, noisy_signal, denoised_signal, metrics, fig)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Run failed: {exc}")


if __name__ == "__main__":
    main()
