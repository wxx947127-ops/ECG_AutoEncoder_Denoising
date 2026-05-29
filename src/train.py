from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ECGDataset
from model import ECGAutoEncoder
from noise import add_noise


BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
SEGMENT_LENGTH = 1000
MAX_FILES = None
METRIC_SAMPLE_COUNT = 100


def plot_loss(train_losses, val_losses, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("训练轮次")
    plt.ylabel("损失")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_denoise_result(clean_signal, noisy_signal, denoised_signal, output_path):
    clean_signal = clean_signal.squeeze().detach().cpu().numpy()
    noisy_signal = noisy_signal.squeeze().detach().cpu().numpy()
    denoised_signal = denoised_signal.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(clean_signal, label="干净信号", linewidth=1.2)
    plt.plot(noisy_signal, label="带噪信号", alpha=0.55, linewidth=1.0)
    plt.plot(denoised_signal, label="去噪后信号", linewidth=1.2)
    plt.xlabel("采样点")
    plt.ylabel("幅值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def denoise(model, noisy_signal):
    predicted_noise = model(noisy_signal)
    denoised_signal = noisy_signal - predicted_noise
    return torch.clamp(denoised_signal, -1.0, 1.0), predicted_noise


def first_difference(signal):
    return signal[..., 1:] - signal[..., :-1]


def denoise_loss(denoised_signal, clean_signal):
    mse_loss = nn.functional.mse_loss(denoised_signal, clean_signal)
    mae_loss = nn.functional.l1_loss(denoised_signal, clean_signal)
    diff_loss = nn.functional.mse_loss(
        first_difference(denoised_signal), first_difference(clean_signal)
    )
    return mse_loss + 0.2 * mae_loss + 0.1 * diff_loss


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


def calculate_quality_metrics(clean_signal, noisy_signal, denoised_signal):
    mse = float(np.mean((denoised_signal - clean_signal) ** 2))
    mae = float(np.mean(np.abs(denoised_signal - clean_signal)))
    pearson = calculate_pearson(clean_signal, denoised_signal)
    before_snr = calculate_snr(clean_signal, noisy_signal)
    after_snr = calculate_snr(clean_signal, denoised_signal)
    snr_improvement = after_snr - before_snr
    return {
        "MSE": mse,
        "MAE": mae,
        "Pearson相关系数": pearson,
        "SNR提升(dB)": snr_improvement,
        "去噪前SNR(dB)": before_snr,
        "去噪后SNR(dB)": after_snr,
    }


def evaluate_random_samples(model, dataset, device, sample_count, output_path):
    model.eval()
    actual_count = min(sample_count, len(dataset))
    generator = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(len(dataset), generator=generator)[:actual_count]

    rows = []
    with torch.no_grad():
        for row_index, dataset_index in enumerate(sample_indices.tolist(), start=1):
            clean_signal = dataset[dataset_index].unsqueeze(0).to(device)
            noisy_signal = add_noise(clean_signal, noise_type="mixed")
            denoised_signal, _ = denoise(model, noisy_signal)

            clean_np = clean_signal.squeeze().detach().cpu().numpy()
            noisy_np = noisy_signal.squeeze().detach().cpu().numpy()
            denoised_np = denoised_signal.squeeze().detach().cpu().numpy()

            metrics = calculate_quality_metrics(clean_np, noisy_np, denoised_np)
            rows.append(
                {
                    "统计类型": "随机样本",
                    "样本编号": row_index,
                    "数据集索引": dataset_index,
                    **metrics,
                }
            )

    metric_columns = ["MSE", "MAE", "Pearson相关系数", "SNR提升(dB)"]
    summary_rows = []
    for summary_name, reducer in (("均值", np.mean), ("标准差", np.std)):
        summary = {"统计类型": summary_name, "样本编号": "", "数据集索引": ""}
        for column in metric_columns:
            values = [row[column] for row in rows]
            summary[column] = float(reducer(values))
        summary_rows.append(summary)

    mean_std_row = {"统计类型": "均值±标准差", "样本编号": "", "数据集索引": ""}
    for column in metric_columns:
        values = np.array([row[column] for row in rows], dtype=np.float64)
        mean_std_row[column] = f"{values.mean():.6f} ± {values.std():.6f}"
    summary_rows.append(mean_std_row)

    pd.DataFrame(rows + summary_rows).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )

    return summary_rows[0], summary_rows[1], mean_std_row, actual_count


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = ECGDataset(
        raw_dir=raw_dir, segment_length=SEGMENT_LENGTH, max_files=MAX_FILES
    )
    if MAX_FILES is None:
        print(f"已从全部CSV文件加载 {len(dataset)} 个片段")
    else:
        print(f"已从前 {MAX_FILES} 个CSV文件加载 {len(dataset)} 个片段")

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"至少需要2个ECG片段才能划分训练集和验证集，当前只有 {len(dataset)} 个"
        )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGAutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    metrics = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for clean_signal in train_loader:
            clean_signal = clean_signal.to(device)
            noisy_signal = add_noise(clean_signal, noise_type="mixed")

            optimizer.zero_grad()
            denoised_signal, _ = denoise(model, noisy_signal)
            loss = denoise_loss(denoised_signal, clean_signal)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * clean_signal.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean_signal in val_loader:
                clean_signal = clean_signal.to(device)
                noisy_signal = add_noise(clean_signal, noise_type="mixed")
                denoised_signal, _ = denoise(model, noisy_signal)
                loss = denoise_loss(denoised_signal, clean_signal)
                val_loss += loss.item() * clean_signal.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        previous_best_val_loss = best_val_loss
        is_best = val_loss < best_val_loss
        has_clear_improvement = previous_best_val_loss - val_loss > MIN_DELTA

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"最佳模型已保存：第 {best_epoch} 轮，验证损失={val_loss:.6f}")

        if has_clear_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "is_best": is_best,
            }
        )

        print(
            f"轮次 [{epoch + 1}/{MAX_EPOCHS}] "
            f"训练损失: {train_loss:.6f} 验证损失: {val_loss:.6f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print("触发早停")
            break

    total_training_epochs = len(metrics)
    best_model_path = results_dir / "best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    plot_loss(train_losses, val_losses, results_dir / "loss.png")
    pd.DataFrame(metrics).rename(
        columns={
            "epoch": "轮次",
            "train_loss": "训练损失",
            "val_loss": "验证损失",
            "is_best": "是否最佳模型",
        }
    ).to_csv(results_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    model.eval()
    with torch.no_grad():
        clean_signal = next(iter(val_loader)).to(device)
        clean_signal = clean_signal[:1]
        noisy_signal = add_noise(clean_signal, noise_type="mixed")
        denoised_signal, _ = denoise(model, noisy_signal)
        plot_denoise_result(
            clean_signal,
            noisy_signal,
            denoised_signal,
            results_dir / "denoise_result.png",
        )

    mean_row, std_row, mean_std_row, metric_count = evaluate_random_samples(
        model,
        val_dataset,
        device,
        METRIC_SAMPLE_COUNT,
        results_dir / "metrics.csv",
    )

    print(f"最佳轮次: {best_epoch}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"实际训练轮数: {total_training_epochs}")
    print(f"最佳模型保存路径: {best_model_path}")
    print(f"损失曲线保存路径: {results_dir / 'loss.png'}")
    print(f"训练历史保存路径: {results_dir / 'training_history.csv'}")
    print(f"去噪效果图保存路径: {results_dir / 'denoise_result.png'}")
    print(f"随机统计样本数: {metric_count}")
    print(f"指标统计保存路径: {results_dir / 'metrics.csv'}")
    for metric_name in ["MSE", "MAE", "Pearson相关系数", "SNR提升(dB)"]:
        print(f"{metric_name}: {mean_std_row[metric_name]}")


if __name__ == "__main__":
    main()
