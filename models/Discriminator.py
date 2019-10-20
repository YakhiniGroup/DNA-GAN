import torch.nn as nn
from models.ResBlock import ResBlock


class Discriminator(nn.Module):
    r"""A convolutional discriminative neural network, with residual layers.
    """
    def __init__(self, conv_dim, seq_length, ):
        super(Discriminator, self).__init__()

        self._conv_dim = conv_dim
        self._seq_length = seq_length
        self.block = nn.Sequential(
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
        )

        # 4 is the size of input channels, as the number of chars in our alphabet.
        self.conv1d = nn.Conv1d(4, conv_dim, 1)
        self.linear = nn.Linear(seq_length * conv_dim, 1)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1,  self._seq_length * self._conv_dim)
        output = self.linear(output)
        return output