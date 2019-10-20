import torch.nn as nn


class WGANGP(nn.Module):
    """A WGAN-GP network, consisting of a discriminator and generator.
    Both could be any nn, as long as generator exposes the latent dim variable.
    """
    def __init__(self, discriminator, generator,):
        super(WGANGP, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
