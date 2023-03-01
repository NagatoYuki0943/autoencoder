import numpy as np
import torch
from torch import nn, Tensor
from torch import optim
import matplotlib.pyplot as plt
from dataset_pre import minst
from tqdm import tqdm

np.random.seed(33)   # random seed，to reproduce results.

CHANNEL_INPUT = 1
CHANNEL_1 = 16
CHANNEL_2 = 8
CHANNEL_OUTPUT = 1
EPOCHS = 20
BATCH_SIZE = 64


class ConvAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(CHANNEL_INPUT, CHANNEL_1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(CHANNEL_1, CHANNEL_2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(CHANNEL_2, CHANNEL_2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(CHANNEL_2, CHANNEL_1, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(CHANNEL_1, CHANNEL_OUTPUT, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x) # [B, 1, 28, 28] -> [B, 8, 7, 7]
        y = self.decoder(x) # [B, 8, 7, 7] -> [B, 1, 28, 28]
        return x, y


if __name__ == "__main__":
    x = torch.ones(1, 1, 28, 28)
    model = ConvAutoEncoder()
    model.eval()
    with torch.inference_mode():
        res = model(x)
    print(res[0].size())    # [B, 8, 7, 7]
    print(res[1].size())    # [B, 1, 28, 28]
