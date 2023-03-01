import numpy as np
import torch
from torch import nn, Tensor
from torch import optim
import matplotlib.pyplot as plt
from dataset_pre import minst
from tqdm import tqdm

np.random.seed(33)   # random seed，to reproduce results.

ENCODING_DIM_INPUT = 784
ENCODING_DIM_LAYER1 = 128
ENCODING_DIM_LAYER2 = 64
ENCODING_DIM_LAYER3 = 10
ENCODING_DIM_OUTPUT = 2
EPOCHS = 20
BATCH_SIZE = 64


class StackAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ENCODING_DIM_INPUT, ENCODING_DIM_LAYER1), # 这一个需要稀疏化
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER1, ENCODING_DIM_LAYER2),
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER2, ENCODING_DIM_LAYER3),
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER3, ENCODING_DIM_OUTPUT),
        )
        self.decoder = nn.Sequential(
            nn.Linear(ENCODING_DIM_OUTPUT, ENCODING_DIM_LAYER3),
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER3, ENCODING_DIM_LAYER2),
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER2, ENCODING_DIM_LAYER1),
            nn.ReLU(),
            nn.Linear(ENCODING_DIM_LAYER1, ENCODING_DIM_INPUT),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x) # [B, 784] -> [B, 2]
        y = self.decoder(x) # [B, 2] -> [B, 784]
        return x, y


if __name__ == "__main__":
    x = torch.ones(1, 784)
    model = StackAutoEncoder()
    model.eval()
    with torch.inference_mode():
        res = model(x)
    print(res[0].size())    # [B, 2]
    print(res[1].size())    # [B, 784]
