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
    "混合噪声": "mixed",
    "高斯噪声": "gaussian",
    "工频干扰": "line",
    "基线漂移": "baseline",
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGAutoEncoder().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def load_ecg_signal(uploaded_file):
    df = pd.read_csv(uploaded_file, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV至少需要2列，当前无法读取第2列ECG信号。")

    signal = df.iloc[:, 1].dropna().to_numpy(dtype=np.float32)
    if len(signal) < SEGMENT_LENGTH:
        raise ValueError(f"ECG信号长度不足{SEGMENT_LENGTH}，无法进行模型推理。")

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
    before_snr = calculate_snr(clean_signal, noisy_signal)
    after_snr = calculate_snr(clean_signal, denoised_signal)
    return {
        "MSE": float(np.mean((denoised_signal - clean_signal) ** 2)),
        "MAE": float(np.mean(np.abs(denoised_signal - clean_signal))),
        "Pearson": calculate_pearson(clean_signal, denoised_signal),
        "SNR Improvement": after_snr - before_snr,
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
        ("原始信号", "原始信号", "green", clean_signal),
        ("带噪信号", "带噪信号", "red", noisy_signal),
        ("去噪信号", "去噪信号", "blue", denoised_signal),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=120)
    x = np.arange(SEGMENT_LENGTH)

    for axis, (title, label, color, signal) in zip(axes, signals):
        axis.plot(x, signal, color=color, linewidth=1.2, label=label)
        draw_annotations(axis, signal, r_peaks, **feature_options)
        axis.set_title(title)
        axis.set_xlim(0, SEGMENT_LENGTH - 1)
        axis.set_ylim(-1.08, 1.08)
        axis.grid(True, alpha=0.5)
        axis.legend(loc="lower left")

    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="ECG去噪系统", layout="wide")
    st.title("ECG心电信号去噪系统")

    with st.sidebar:
        st.header("参数设置")
        uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
        noise_label = st.selectbox("噪声类型", list(NOISE_TYPE_MAP.keys()), index=0)

        st.header("波形特征显示")
        show_p = st.checkbox("显示P波", value=False)
        show_qrs = st.checkbox("显示QRS波群", value=False)
        show_t = st.checkbox("显示T波", value=False)
        show_r = st.checkbox("显示R峰", value=False)

    if uploaded_file is None:
        st.info("请上传无表头、12列的ECG CSV文件。系统默认读取第2列作为单导联ECG信号。")
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
        metric_cols[3].metric("SNR Improvement", f"{metrics['SNR Improvement']:.2f} dB")

        feature_options = {
            "show_p": show_p,
            "show_qrs": show_qrs,
            "show_t": show_t,
            "show_r": show_r,
        }
        fig = plot_signals(clean_signal, noisy_signal, denoised_signal, feature_options)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.error(f"运行失败: {exc}")


if __name__ == "__main__":
    main()
