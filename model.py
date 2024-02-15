import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, num_dimensions):
        super(LogisticRegression, self).__init__()

        # self.linear = nn.Linear(num_dimensions, 1, bias=True)
        self.theta = nn.Parameter(torch.zeros(num_dimensions, 1))

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.theta))