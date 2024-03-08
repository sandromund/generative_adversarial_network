import math

import torch
from torch.utils.data import DataLoader


def get_demo_data_loader(batch_size: int, train_data_length=1024) -> DataLoader:
    """
    import matplotlib.pyplot as plt
    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    """
    train_data = torch.zeros((train_data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length)
    train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
