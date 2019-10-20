import torch.nn as nn
from models.ResBlock import ResBlock


class Generator(nn.Module):
    r"""A convolutional generative neural network, with residual layers.
        Must expose a latent_dim variable.
    """
    def __init__(self, latent_dim, conv_dim, seq_length):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self._conv_dim = conv_dim
        self._seq_length = seq_length

        self.fc1 = nn.Linear(latent_dim, conv_dim * seq_length)
        self.block = nn.Sequential(
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
            ResBlock(conv_dim, 0.3),
        )
        self.conv1 = nn.Conv1d(conv_dim, 4, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, noise):
        batch_size = noise.size()[0]
        output = self.fc1(noise)
        output = output.view(-1, self._conv_dim, self._seq_length)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size * self._seq_length, -1)
        output = self.softmax(output)
        return output.view(shape)