from io import BytesIO
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.signal import butter, filtfilt, find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import ECGAutoEncoder
from noise import add_noise


SEGMENT_LENGTH = 1000
MODEL_PATH = PROJECT_ROOT / "results" / "best_model.pth"
SAMPLING_RATE = 500
BUTTERWORTH_CUTOFF = 40
BUTTERWORTH_ORDER = 4

TXT = {
    "title": "\u0045\u0043\u0047\u5fc3\u7535\u4fe1\u53f7\u53bb\u566a\u7cfb\u7edf",
    "settings": "\u53c2\u6570\u8bbe\u7f6e",
    "upload": "\u4e0a\u4f20\u0043\u0053\u0056\u6587\u4ef6",
    "noise_type": "\u566a\u58f0\u7c7b\u578b",
    "mixed": "\u6df7\u5408\u566a\u58f0",
    "gaussian": "\u9ad8\u65af\u566a\u58f0",
    "line": "\u5de5\u9891\u5e72\u6270",
    "baseline": "\u57fa\u7ebf\u6f02\u79fb",
    "features": "\u6ce2\u5f62\u7279\u5f81\u663e\u793a",
    "show_p": "\u663e\u793a\u0050\u6ce2",
    "show_qrs": "\u663e\u793a\u0051\u0052\u0053\u6ce2\u7fa4",
    "show_t": "\u663e\u793a\u0054\u6ce2",
    "show_r": "\u663e\u793a\u0052\u5cf0",
    "info": "\u8bf7\u4e0a\u4f20\u65e0\u8868\u5934\u3001\u0031\u0032\u5217\u7684\u0045\u0043\u0047\u0020\u0043\u0053\u0056\u6587\u4ef6\u3002\u7cfb\u7edf\u9ed8\u8ba4\u8bfb\u53d6\u7b2c\u0032\u5217\u4f5c\u4e3a\u5355\u5bfc\u8054\u0045\u0043\u0047\u4fe1\u53f7\u3002",
    "comparison": "\u53bb\u566a\u65b9\u6cd5\u5bf9\u6bd4",
    "download": "\u4e0b\u8f7d\u7ed3\u679c",
    "download_signal": "\u4e0b\u8f7d denoised_signal.csv",
    "download_metrics": "\u4e0b\u8f7d metrics.csv",
    "download_png": "\u4e0b\u8f7d result.png",
    "run_failed": "\u8fd0\u884c\u5931\u8d25",
}

NOISE_TYPE_MAP = {
    TXT["mixed"]: "mixed",
    TXT["gaussian"]: "gaussian",
    TXT["line"]: "line",
    TXT["baseline"]: "baseline",
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


def butterworth_lowpass_filter(signal, sampling_rate=SAMPLING_RATE, cutoff=BUTTERWORTH_CUTOFF, order=BUTTERWORTH_ORDER):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")
    filtered_signal = filtfilt(b, a, signal)
    return np.clip(filtered_signal, -1.0, 1.0).astype(np.float32)


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


def make_comparison_df(autoencoder_metrics, butterworth_metrics):
    return pd.DataFrame(
        [
            {
                "Method": "AutoEncoder",
                "MSE": autoencoder_metrics["MSE"],
                "MAE": autoencoder_metrics["MAE"],
                "Pearson": autoencoder_metrics["Pearson"],
                "SNR Improvement": autoencoder_metrics["SNR_improvement"],
            },
            {
                "Method": "Butterworth",
                "MSE": butterworth_metrics["MSE"],
                "MAE": butterworth_metrics["MAE"],
                "Pearson": butterworth_metrics["Pearson"],
                "SNR Improvement": butterworth_metrics["SNR_improvement"],
            },
        ]
    )


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


def plot_signals(clean_signal, noisy_signal, denoised_signal, butterworth_signal, feature_options):
    r_peaks = detect_r_peaks(clean_signal)
    signals = [
        ("Original Signal", "Original Signal", "green", clean_signal),
        ("Noisy Signal", "Noisy Signal", "red", noisy_signal),
        ("Denoised Signal", "Denoised Signal", "blue", denoised_signal),
        ("Butterworth Filtered Signal", "Butterworth Filtered Signal", "purple", butterworth_signal),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=120)
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


def make_signal_csv(clean_signal, noisy_signal, denoised_signal, butterworth_signal):
    signal_df = pd.DataFrame(
        {
            "clean_signal": clean_signal,
            "noisy_signal": noisy_signal,
            "denoised_signal": denoised_signal,
            "butterworth_signal": butterworth_signal,
        }
    )
    return signal_df.to_csv(index=False).encode("utf-8-sig")


def make_metrics_csv(comparison_df):
    return comparison_df.to_csv(index=False).encode("utf-8-sig")


def make_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def show_download_area(clean_signal, noisy_signal, denoised_signal, butterworth_signal, comparison_df, fig):
    st.subheader(TXT["download"])
    download_cols = st.columns(3)
    download_cols[0].download_button(
        label=TXT["download_signal"],
        data=make_signal_csv(clean_signal, noisy_signal, denoised_signal, butterworth_signal),
        file_name="denoised_signal.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        label=TXT["download_metrics"],
        data=make_metrics_csv(comparison_df),
        file_name="metrics.csv",
        mime="text/csv",
    )
    download_cols[2].download_button(
        label=TXT["download_png"],
        data=make_png(fig),
        file_name="result.png",
        mime="image/png",
    )


def main():
    st.set_page_config(page_title=TXT["title"], layout="wide")
    st.title(TXT["title"])

    with st.sidebar:
        st.header(TXT["settings"])
        uploaded_file = st.file_uploader(TXT["upload"], type=["csv"])
        noise_label = st.selectbox(TXT["noise_type"], list(NOISE_TYPE_MAP.keys()), index=0)

        st.header(TXT["features"])
        show_p = st.checkbox(TXT["show_p"], value=False)
        show_qrs = st.checkbox(TXT["show_qrs"], value=False)
        show_t = st.checkbox(TXT["show_t"], value=False)
        show_r = st.checkbox(TXT["show_r"], value=False)

    if uploaded_file is None:
        st.info(TXT["info"])
        return

    try:
        model, device = load_model()
        clean_signal = load_ecg_signal(uploaded_file)
        noisy_signal, denoised_signal = run_autoencoder(
            model, device, clean_signal, NOISE_TYPE_MAP[noise_label]
        )
        butterworth_signal = butterworth_lowpass_filter(noisy_signal)

        autoencoder_metrics = calculate_metrics(clean_signal, noisy_signal, denoised_signal)
        butterworth_metrics = calculate_metrics(clean_signal, noisy_signal, butterworth_signal)
        comparison_df = make_comparison_df(autoencoder_metrics, butterworth_metrics)

        metric_cols = st.columns(4)
        metric_cols[0].metric("MSE", f"{autoencoder_metrics['MSE']:.6f}")
        metric_cols[1].metric("MAE", f"{autoencoder_metrics['MAE']:.6f}")
        metric_cols[2].metric("Pearson", f"{autoencoder_metrics['Pearson']:.4f}")
        metric_cols[3].metric("SNR Improvement", f"{autoencoder_metrics['SNR_improvement']:.2f} dB")

        st.subheader(TXT["comparison"])
        st.dataframe(
            comparison_df.style.format(
                {
                    "MSE": "{:.6f}",
                    "MAE": "{:.6f}",
                    "Pearson": "{:.4f}",
                    "SNR Improvement": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        feature_options = {
            "show_p": show_p,
            "show_qrs": show_qrs,
            "show_t": show_t,
            "show_r": show_r,
        }
        fig = plot_signals(
            clean_signal,
            noisy_signal,
            denoised_signal,
            butterworth_signal,
            feature_options,
        )
        st.pyplot(fig, clear_figure=False)
        show_download_area(
            clean_signal,
            noisy_signal,
            denoised_signal,
            butterworth_signal,
            comparison_df,
            fig,
        )
        plt.close(fig)
    except Exception as exc:
        st.error(f"{TXT['run_failed']}: {exc}")


if __name__ == "__main__":
    main()
