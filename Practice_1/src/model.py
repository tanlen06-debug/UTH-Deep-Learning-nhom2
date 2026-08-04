import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    MLP model for FashionMNIST classification.

    Input shape:
        [batch_size, 1, 28, 28]

    Output shape:
        [batch_size, 10]
    """

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits