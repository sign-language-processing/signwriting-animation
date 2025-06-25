import torch
from torch import nn

class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = logvar
        self.std = torch.exp(0.5 * logvar)
        self.distribution = torch.distributions.Normal(self.mean, self.std)

    def sample(self):
        return self.distribution.rsample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def nll(self, value):
        return -self.log_prob(value).mean()
    
    @property
    def stddev(self):
        return self.std 

class DistributionPredictionModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # distribution.py
        self.fc_mu = nn.Sequential(nn.Linear(input_size, 1), nn.Softplus())
        self.fc_var = nn.Sequential(nn.Linear(input_size, 1), nn.Softplus(), nn.Hardtanh(min_val=1e-4, max_val=1.0))

    def forward(self, x: torch.Tensor):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return DiagonalGaussianDistribution(mu, log_var)