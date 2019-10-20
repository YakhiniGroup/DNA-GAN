import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, dim, r):
        super(ResBlock, self).__init__()

        self.r = r
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(dim, dim, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (self.r * output)