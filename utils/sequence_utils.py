import collections
import random

import numpy as np
import torch
from itertools import tee, islice


def sequence_to_one_hot(sequence: list) -> list:
    """Gets a sequence of size L and creates a matrix of LX4 of one hot encoding of each vector.
    Args:
        sequence (list): a list of letters out of the alphabet 'acgt'

    Returns:
        One hot encoding of the sequence, such that a = [1,0,0,0] ,c = [0,1,0,0]
        g = [0,0,1,0], t = [0,0,0,1]
    """
    alphabet = 'acgt'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in sequence]
    # one hot encode
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


def idx_to_char(idx_tensor):
    return "acgt"[idx_tensor.item()]


def one_hot_to_seq(one_hot_vector):
    return ''.join(list(map(idx_to_char, one_hot_vector)))


def to_one_hot(vector):
    if type(vector) == np.ndarray:
        vector.argmax(axis=1)

    elif type(vector) == torch.Tensor:
        return vector.max(dim=1)[1]


def vector_to_seq(vector):
    return one_hot_to_seq(to_one_hot(vector))


def batch_to_one_hot(batch: torch.Tensor) -> torch.Tensor:
    r"""Turn a batch of probabilities into a one hot vector.

    Args:
        batch (torch.Tensor): A batch of sequences we wish to calculate kmer proportions in.

    Returns:
        torch.Tensor
    """
    batch_size = batch.size()[0]
    sequence_length = batch.size()[1]

    argmax = batch.max(dim=2)[1]
    one_hot = torch.zeros_like(batch)

    # Can probably use unsqueeze instead, for now its ok.
    one_hot[np.arange(batch_size).reshape(batch_size, -1),
            np.arange(sequence_length).reshape(-1, sequence_length), argmax] = 1
    one_hot = torch.where(one_hot == 1, one_hot, torch.zeros_like(one_hot))
    return one_hot


def generate_random_sequence(length):
    return ''.join(random.choices('acgt', k=length))


def sample_batch(batch_size: int, sequence_length: int) -> torch.Tensor:
    r"""Generates a random batch of batch_size sequences of length sequence_length

    Args:
        batch_size: The size of the batch we wish to generate
        sequence_length: Length of each sequence in the batch

    Returns:
         One-hot torch.Tensor of shape (batch_size,sequence_length,4),
         Such on dim 2 every vector contains 3 0's and a 1.
    """
    # choose randomly the indexs which will contain ones
    rand_idxs = torch.randint(0, 4, (batch_size, sequence_length), dtype=torch.long)
    rand_idxs = torch.unsqueeze(rand_idxs, 2)
    one_hot = torch.zeros(batch_size, sequence_length, 4)
    # plant ones at rand idxs. note that scatter_ operates inplace
    # 1 is broadcasted to rand_idxs
    one_hot.scatter_(2, rand_idxs, 1)
    return one_hot


def calc_base_proportions(batch: torch.Tensor) -> torch.Tensor:
    r"""Calculate the proportions of nucleotide basis in a given batch

    Args:
        batch: A batch sequences we wish to calculate base proportions in.

    Returns:
        torch.Tensor of shape (4) such that with proportions of acgt in this order.
    """
    batch_size = batch.size()[0]
    sequence_length = batch.size()[1]

    one_hot = batch_to_one_hot(batch)

    return one_hot.sum(dim=(0, 1)) / (batch_size * sequence_length)


def _find_kmers_and_proportions_iterative(batch: torch.Tensor, k: int)\
        -> (torch.Tensor, torch.Tensor):
    r"""Calculate the proportions of nucleotide basis in a given batch in iterative manner

    Args:
        batch (torch.Tensor): A batch of sequencess we wish to calculate kmer proportions in.
        k (int): the size of kmers.

    Returns:
        (torch.Tensor,torch.Tensor) - tuple of tensors:
            tensor of shape (num of kmers, k, 4) all kmers in one hot form,
            tensor of shape (num of kmers) with proportion of each kmer.
    """
    # reshape batch to be batch of basis
    batch = batch.reshape(-1, 4)
    iters = tee(batch, k)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)

    kmer_counts = collections.defaultdict(int)
    total_kmers = 0
    for kmer in zip(*iters):
        if k == 1:
            kmer_counts[vector_to_seq(kmer.unsqueeze(dim=0))] += 1
        else:
            kmer_counts[vector_to_seq(kmer)] += 1
        total_kmers += 1

    return kmer_counts, total_kmers


def _find_kmers_and_proportions_vectorized(batch: torch.Tensor, k: int)\
        -> (torch.Tensor, torch.Tensor):
    r"""Calculate the proportions of nucleotide basis in a given batch in vectorized manner

    Args:
        batch (torch.Tensor): A batch of sequencess we wish to calculate kmer proportions in.
        k (int): the size of kmers.

    Returns:
        (torch.Tensor,torch.Tensor) - tuple of tensors:
            tensor of shape (num of kmers, k, 4) all kmers in one hot form,
            tensor of shape (num of kmers) with proportion of each kmer.
    """
    # TODO - Before going to testing, need to change this to torch only, so we can work entierly on gpu.
    # Work with numpy, as torch is missing roll and unique (with vectors) functions.
    # this might not be efficient on gpu, for now ok.
    # We create a copy of the batch, as we don't wan to change it.
    batch_as_np = batch.detach().cpu().numpy()
    batch_as_np = batch_as_np.reshape(-1, 4)

    # If k == 1, then no need to concatenate any array.
    kmers_with_repeats = batch_as_np
    # Else,
    if k != 1:
        array_of_shifted = [np.roll(batch_as_np, i, 0) for i in range(k-1, -1, -1)]
        # Notice takes O(batch) memory
        kmers_with_repeats = np.concatenate(array_of_shifted, axis=1)[k-1:]

    kmers, counts = np.unique(kmers_with_repeats, return_counts=True, axis=0)
    # unique returns a double array instead of float for counts.
    counts = counts.astype(np.float32)

    # reshape back
    kmers = kmers.reshape(-1, k, 4)

    num_kmers = batch_as_np.shape[0] + k + 1
    proportions = counts / num_kmers

    # TODO - Maybe we should just leave as numpy, not do any of this on gpu?
    return torch.from_numpy(kmers), torch.from_numpy(proportions)


def find_kmers_and_proportions(batch: torch.Tensor, k: int, vectorized=True)\
        -> (torch.Tensor, torch.Tensor):
    r"""Calculate the proportions of nucleotide basis in a given batch in iterative manner

    Args:
        batch (torch.Tensor): A batch of sequences we wish to calculate kmer proportions in.
        k (int): the size of kmers.
        vectorized (bool): Use only for small batches, might cause performance improvement

    Returns:
        (torch.Tensor,torch.Tensor) - tuple of tensors:
            tensor of shape (num of kmers, k, 4) all kmers in one hot form,
            tensor of shape (num of kmers) with proportion of each kmer.
    """
    batch = batch_to_one_hot(batch)

    if vectorized:
        return _find_kmers_and_proportions_vectorized(batch, k)
    else:
        _find_kmers_and_proportions_iterative(batch, k)