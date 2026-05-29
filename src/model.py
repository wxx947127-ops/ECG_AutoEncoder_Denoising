import torch.nn as nn


class ECGAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                32, 16, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                16, 1, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        predicted_noise = 0.3 * x[..., :1000]
        return predicted_noise
