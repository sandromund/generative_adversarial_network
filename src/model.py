from torch import nn

from const import LATENT_SPACE_SAMPLE, DATA_DIMENSION, ONE_HOT_ENCODING


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(DATA_DIMENSION, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), DATA_DIMENSION)
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_SPACE_SAMPLE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, DATA_DIMENSION),
        )

    def forward(self, x):
        output = self.model(x)
        if ONE_HOT_ENCODING:
            output = output.view(x.size(0), 36, 36, 4)
        else:
            output = output.view(x.size(0), 36, 36)
        return output
