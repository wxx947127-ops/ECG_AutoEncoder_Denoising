from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Button, Entry, Frame, Label, LabelFrame, Tk
from tkinter import StringVar
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

from model import ECGAutoEncoder
from noise import add_noise
from traditional_filters import combined_filter


SEGMENT_LENGTH = 1000
NOISE_TYPE_MAP = {
    "基线漂移": "baseline",
    "高斯噪声": "gaussian",
    "工频干扰": "line",
    "混合噪声": "mixed",
}
FEATURE_NAMES = ("P波", "QRS波群", "T波", "R峰")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class ECGDenoiseGUI:
    def __init__(self, root):
        self.root = root
        self.project_root = Path(__file__).resolve().parents[1]
        self.raw_dir = self.project_root / "data" / "raw"
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self._find_model_path()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

        self.clean_signal = None
        self.noisy_signal = None
        self.auto_signal = None
        self.traditional_signal = None
        self.current_signals = None
        self.r_peaks = np.array([], dtype=np.int64)
        self.feature_flags = {feature_name: False for feature_name in FEATURE_NAMES}

        self.root.title("ECG心电信号去噪系统")
        self.root.geometry("1160x860")
        self.root.minsize(1020, 760)

        self.file_path_var = StringVar()
        self.noise_type_var = StringVar(value="混合噪声")
        self.ae_mse_var = StringVar(value="自编码器 MSE: --")
        self.ae_mae_var = StringVar(value="自编码器 MAE: --")
        self.ae_pearson_var = StringVar(value="自编码器 Pearson: --")
        self.ae_snr_var = StringVar(value="自编码器 SNR提升: -- dB")
        self.tf_mse_var = StringVar(value="传统滤波 MSE: --")
        self.tf_mae_var = StringVar(value="传统滤波 MAE: --")
        self.tf_pearson_var = StringVar(value="传统滤波 Pearson: --")
        self.tf_snr_var = StringVar(value="传统滤波 SNR提升: -- dB")

        self._set_default_csv()
        self._build_layout()
        self._draw_empty_plots()

    def _find_model_path(self):
        best_model_path = self.project_root / "results" / "best_model.pth"
        if best_model_path.exists():
            return best_model_path
        return self.project_root / "results" / "model.pth"

    def _load_model(self):
        model = ECGAutoEncoder().to(self.device)
        if not self.model_path.exists():
            return model.eval()

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.eval()

    def _set_default_csv(self):
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if csv_files:
            self.file_path_var.set(str(csv_files[0]))

    def _build_layout(self):
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        left_panel = Frame(main_frame, width=330)
        left_panel.pack(side=LEFT, fill="y", padx=(0, 12))
        left_panel.pack_propagate(False)

        right_panel = Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        settings = LabelFrame(left_panel, text="参数设置", padx=10, pady=10)
        settings.pack(fill=X)

        Label(settings, text="选择ECG数据文件:").pack(anchor="center", pady=(4, 6))
        Entry(settings, textvariable=self.file_path_var).pack(fill=X)
        Button(settings, text="浏览", width=14, command=self.choose_file).pack(pady=8)

        Label(settings, text="噪声类型:").pack(anchor="center", pady=(8, 6))
        noise_combo = ttk.Combobox(
            settings,
            textvariable=self.noise_type_var,
            values=list(NOISE_TYPE_MAP.keys()),
            state="readonly",
            width=18,
        )
        noise_combo.pack()

        Button(settings, text="开始去噪", width=16, command=self.run_denoise).pack(pady=(18, 6))
        Button(settings, text="传统滤波对比", width=16, command=self.run_filter_comparison).pack(pady=6)

        feature_box = LabelFrame(left_panel, text="波形特征标注", padx=10, pady=10)
        feature_box.pack(fill=X, pady=(8, 0))
        for feature_name in FEATURE_NAMES:
            Button(
                feature_box,
                text=f"显示{feature_name}",
                width=16,
                command=lambda name=feature_name: self.toggle_feature(name),
            ).pack(pady=3)

        metrics_box = LabelFrame(left_panel, text="指标对比", padx=10, pady=10)
        metrics_box.pack(fill=X, pady=(16, 0))
        for metric_var in (
            self.ae_mse_var,
            self.ae_mae_var,
            self.ae_pearson_var,
            self.ae_snr_var,
            self.tf_mse_var,
            self.tf_mae_var,
            self.tf_pearson_var,
            self.tf_snr_var,
        ):
            Label(metrics_box, textvariable=metric_var, anchor="w").pack(fill=X, pady=3)

        self.figure, self.axes = plt.subplots(4, 1, figsize=(8, 8), dpi=100)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=self.raw_dir,
            title="选择ECG数据文件",
            filetypes=(("CSV文件", "*.csv"), ("所有文件", "*.*")),
        )
        if file_path:
            self.file_path_var.set(file_path)

    def toggle_feature(self, feature_name):
        if self.current_signals is None:
            messagebox.showinfo("提示", "请先选择ECG文件并完成去噪")
            return

        self.feature_flags[feature_name] = not self.feature_flags[feature_name]
        self._draw_signals(self.current_signals)

    def run_denoise(self):
        try:
            self.clean_signal = self._load_clean_signal(Path(self.file_path_var.get()))
            clean_tensor = (
                torch.from_numpy(self.clean_signal).float().view(1, 1, SEGMENT_LENGTH)
            ).to(self.device)

            noise_type = NOISE_TYPE_MAP[self.noise_type_var.get()]
            noisy_tensor = add_noise(clean_tensor, noise_type)
            self.noisy_signal = noisy_tensor.squeeze().detach().cpu().numpy()
            self.auto_signal = self._autoencoder_denoise(noisy_tensor)
            self.traditional_signal = None

            self._set_auto_metrics(self._calculate_metrics(self.clean_signal, self.noisy_signal, self.auto_signal))
            self._clear_traditional_metrics()

            self.current_signals = [
                ("原始干净信号", "干净信号", "green", self.clean_signal),
                ("带噪声信号", "带噪信号", "red", self.noisy_signal),
                ("自编码器去噪结果", "自编码器", "blue", self.auto_signal),
            ]
            self.r_peaks = self._detect_r_peaks(self.clean_signal)
            self._draw_signals(self.current_signals)
        except Exception as exc:
            messagebox.showerror("运行失败", str(exc))

    def run_filter_comparison(self):
        if self.clean_signal is None or self.noisy_signal is None or self.auto_signal is None:
            messagebox.showinfo("提示", "请先选择ECG文件并完成去噪")
            return

        try:
            self.traditional_signal = combined_filter(self.noisy_signal)
            ae_metrics = self._calculate_metrics(self.clean_signal, self.noisy_signal, self.auto_signal)
            tf_metrics = self._calculate_metrics(
                self.clean_signal, self.noisy_signal, self.traditional_signal
            )
            self._set_auto_metrics(ae_metrics)
            self._set_traditional_metrics(tf_metrics)

            self.current_signals = [
                ("原始干净信号", "干净信号", "green", self.clean_signal),
                ("带噪声信号", "带噪信号", "red", self.noisy_signal),
                ("自编码器去噪结果", "自编码器", "blue", self.auto_signal),
                ("传统滤波结果", "传统滤波", "purple", self.traditional_signal),
            ]
            self._draw_signals(self.current_signals)
            self._save_filter_comparison(ae_metrics, tf_metrics)
        except Exception as exc:
            messagebox.showerror("传统滤波对比失败", str(exc))

    def _autoencoder_denoise(self, noisy_tensor):
        with torch.no_grad():
            predicted_noise = self.model(noisy_tensor)
            denoised_tensor = torch.clamp(noisy_tensor - predicted_noise, -1.0, 1.0)
        return denoised_tensor.squeeze().detach().cpu().numpy()

    def _save_filter_comparison(self, ae_metrics, tf_metrics):
        rows = [
            {"方法": "自编码器", **ae_metrics},
            {"方法": "传统滤波", **tf_metrics},
        ]
        pd.DataFrame(rows).to_csv(
            self.results_dir / "filter_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.figure.savefig(self.results_dir / "filter_comparison.png", dpi=150)

    def _load_clean_signal(self, file_path):
        if not file_path.exists():
            raise FileNotFoundError("请选择有效的CSV文件")

        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            raise ValueError("CSV至少需要2列，当前无法提取B列")

        signal = df.iloc[:, 1].dropna().to_numpy(dtype=np.float32)
        if signal.shape[0] < SEGMENT_LENGTH:
            raise ValueError(f"信号长度不足{SEGMENT_LENGTH}，无法显示去噪结果")

        signal = signal[:SEGMENT_LENGTH]
        signal_min = float(signal.min())
        signal_max = float(signal.max())
        if signal_max > signal_min:
            signal = 2.0 * (signal - signal_min) / (signal_max - signal_min) - 1.0
        else:
            signal = np.zeros_like(signal, dtype=np.float32)
        return signal.astype(np.float32)

    def _detect_r_peaks(self, signal):
        prominence = max(0.15, float(np.std(signal)) * 0.8)
        peaks, _ = find_peaks(signal, distance=150, prominence=prominence)
        if len(peaks) == 0:
            height = float(np.percentile(signal, 90))
            peaks, _ = find_peaks(signal, distance=150, height=height)
        return peaks.astype(np.int64)

    def _calculate_metrics(self, clean_signal, noisy_signal, denoised_signal):
        before_snr = self._calculate_snr(clean_signal, noisy_signal)
        after_snr = self._calculate_snr(clean_signal, denoised_signal)
        return {
            "MSE": float(np.mean((denoised_signal - clean_signal) ** 2)),
            "MAE": float(np.mean(np.abs(denoised_signal - clean_signal))),
            "Pearson": self._calculate_pearson(clean_signal, denoised_signal),
            "SNR提升(dB)": after_snr - before_snr,
        }

    def _set_auto_metrics(self, metrics):
        self.ae_mse_var.set(f"自编码器 MSE: {metrics['MSE']:.6f}")
        self.ae_mae_var.set(f"自编码器 MAE: {metrics['MAE']:.6f}")
        self.ae_pearson_var.set(f"自编码器 Pearson: {metrics['Pearson']:.4f}")
        self.ae_snr_var.set(f"自编码器 SNR提升: {metrics['SNR提升(dB)']:.2f} dB")

    def _set_traditional_metrics(self, metrics):
        self.tf_mse_var.set(f"传统滤波 MSE: {metrics['MSE']:.6f}")
        self.tf_mae_var.set(f"传统滤波 MAE: {metrics['MAE']:.6f}")
        self.tf_pearson_var.set(f"传统滤波 Pearson: {metrics['Pearson']:.4f}")
        self.tf_snr_var.set(f"传统滤波 SNR提升: {metrics['SNR提升(dB)']:.2f} dB")

    def _clear_traditional_metrics(self):
        self.tf_mse_var.set("传统滤波 MSE: --")
        self.tf_mae_var.set("传统滤波 MAE: --")
        self.tf_pearson_var.set("传统滤波 Pearson: --")
        self.tf_snr_var.set("传统滤波 SNR提升: -- dB")

    @staticmethod
    def _calculate_snr(clean_signal, compared_signal):
        noise = clean_signal - compared_signal
        signal_power = float(np.mean(clean_signal**2))
        noise_power = float(np.mean(noise**2))
        if noise_power <= 1e-12:
            return 99.0
        return 10.0 * np.log10(signal_power / noise_power)

    @staticmethod
    def _calculate_pearson(clean_signal, compared_signal):
        clean_std = float(np.std(clean_signal))
        compared_std = float(np.std(compared_signal))
        if clean_std <= 1e-12 or compared_std <= 1e-12:
            return 0.0
        return float(np.corrcoef(clean_signal, compared_signal)[0, 1])

    def _draw_empty_plots(self):
        zero = np.zeros(SEGMENT_LENGTH)
        self._draw_signals(
            [
                ("原始干净信号", "干净信号", "green", zero),
                ("带噪声信号", "带噪信号", "red", zero),
                ("自编码器去噪结果", "自编码器", "blue", zero),
            ]
        )

    def _draw_signals(self, signal_specs):
        x = np.arange(SEGMENT_LENGTH)

        for axis in self.axes:
            axis.clear()
            axis.set_visible(False)

        for axis, (title, label, color, signal) in zip(self.axes, signal_specs):
            axis.set_visible(True)
            axis.plot(x, signal, color=color, linewidth=1.4, label=label)
            self._draw_feature_annotations(axis, signal)
            axis.set_title(title, fontsize=12)
            axis.grid(True, alpha=0.55)
            axis.legend(loc="lower left", fontsize=9)
            axis.set_xlim(0, SEGMENT_LENGTH - 1)
            axis.set_ylim(-1.08, 1.08)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def _draw_feature_annotations(self, axis, signal):
        if len(self.r_peaks) == 0:
            return

        if self.feature_flags["P波"]:
            self._draw_peak_windows(axis, self.r_peaks, -200, -120, "green", "P", 0.58)
        if self.feature_flags["QRS波群"]:
            self._draw_peak_windows(axis, self.r_peaks, -20, 20, "red", "QRS", 0.84)
        if self.feature_flags["T波"]:
            self._draw_peak_windows(axis, self.r_peaks, 100, 250, "blue", "T", 0.68)
        if self.feature_flags["R峰"]:
            valid_peaks = self.r_peaks[(self.r_peaks >= 0) & (self.r_peaks < len(signal))]
            axis.scatter(
                valid_peaks,
                signal[valid_peaks],
                color="red",
                s=28,
                marker="o",
                zorder=5,
            )
            for peak in valid_peaks:
                text_y = min(float(signal[peak]) + 0.1, 1.02)
                axis.text(peak, text_y, "R", color="red", fontsize=9, ha="center")

    @staticmethod
    def _draw_peak_windows(axis, peaks, start_offset, end_offset, color, label, text_y):
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


def main():
    root = Tk()
    ECGDenoiseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
