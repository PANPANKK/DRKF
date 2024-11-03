import torch
import torch.nn as nn

class TemperatureModel(nn.Module):
    def __init__(self, initial_temp=1.0):
        super(TemperatureModel, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temp))

    def forward(self):
        return torch.sigmoid(self.temperature)  # Limit temperature between 0 and 1
