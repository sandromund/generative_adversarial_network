import torch
from torch import nn

from const import LATENT_SPACE_SAMPLE, DATA_DIMENSION, ONE_HOT_ENCODING


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(DATA_DIMENSION, 648),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(648, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), DATA_DIMENSION)
        output = self.model(x)
        output = torch.round(output)
        return output


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_SPACE_SAMPLE, 648),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(648, 648),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(648, 648),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(648, DATA_DIMENSION),
        )

    def forward(self, x):
        output = self.model(x)
        if ONE_HOT_ENCODING:
            output = output.view(x.size(0), 36, 36, 4)
        else:
            output = output.view(x.size(0), 36, 36)
        return output
