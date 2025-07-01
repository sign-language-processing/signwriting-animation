import torch
from torch import nn

class DistributionPredictionModel(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        self.fc_mu = nn.Linear(input_size, 1)
        self.fc_var = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        return q