import torch


def _random_uniform(low, high, device, dtype):
    return torch.empty(1, device=device, dtype=dtype).uniform_(low, high).item()


def add_gaussian_noise(clean_signal, noise_std=None):
    if noise_std is None:
        noise_std = _random_uniform(0.03, 0.08, clean_signal.device, clean_signal.dtype)
    noise = torch.randn_like(clean_signal) * noise_std
    noisy_signal = clean_signal + noise
    return torch.clamp(noisy_signal, -1.0, 1.0)


def add_line_noise(clean_signal, amplitude=None):
    if amplitude is None:
        amplitude = _random_uniform(0.02, 0.05, clean_signal.device, clean_signal.dtype)
    length = clean_signal.shape[-1]
    t = torch.linspace(
        0.0, 1.0, steps=length, device=clean_signal.device, dtype=clean_signal.dtype
    )
    line_noise = amplitude * torch.sin(2.0 * torch.pi * 50.0 * t)
    while line_noise.dim() < clean_signal.dim():
        line_noise = line_noise.unsqueeze(0)
    return torch.clamp(clean_signal + line_noise, -1.0, 1.0)


def add_baseline_noise(clean_signal, amplitude=None):
    if amplitude is None:
        amplitude = _random_uniform(0.03, 0.08, clean_signal.device, clean_signal.dtype)
    length = clean_signal.shape[-1]
    t = torch.linspace(
        0.0, 1.0, steps=length, device=clean_signal.device, dtype=clean_signal.dtype
    )
    baseline_noise = amplitude * torch.sin(2.0 * torch.pi * 0.8 * t)
    while baseline_noise.dim() < clean_signal.dim():
        baseline_noise = baseline_noise.unsqueeze(0)
    return torch.clamp(clean_signal + baseline_noise, -1.0, 1.0)


def add_noise(clean_signal, noise_type="gaussian"):
    noise_type = noise_type.lower()
    if noise_type == "gaussian":
        return add_gaussian_noise(clean_signal)
    if noise_type == "line":
        return add_line_noise(clean_signal)
    if noise_type == "baseline":
        return add_baseline_noise(clean_signal)
    if noise_type == "mixed":
        noisy_signal = add_gaussian_noise(clean_signal)
        noisy_signal = add_line_noise(noisy_signal)
        return add_baseline_noise(noisy_signal)
    raise ValueError(f"Unknown noise type: {noise_type}")
