from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class EnhancersDataset(Dataset):
    """ An enhancers dataset.
        The dataset set is build by sampling a dataset of enhancers along their intensity values.
    """

    def __init__(self, length):
        """

        Args:
            length: The size of the desired dataset.
        """

        # Get enhancers dataset containig binding sites in the first column
        # and intensities level on the second columns
        enhancers_df = pd.read_csv('./datasets/sequences_intensity.csv')

        # Get sequences and intensities.
        sequences = enhancers_df[enhancers_df.columns[1]]
        intensities = enhancers_df[enhancers_df.columns[2]]

        # Sample sequences.
        self.data = np.random.choice(sequences, size=length, p=intensities)

        # Converts sequences to one-hot.
        self.data = list(map(sequence_to_one_hot, self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx])).float()


def sequence_to_one_hot(sequence: list) -> list:
    """Gets a sequence of size L and creates a matrix of LX4 of one hot encoding of each vector.
    Args:
        sequence (list): a list of letters out of the alphabet 'acgt'

    Returns:
        One hot encoding of the sequence, such that a = [1,0,0,0] ,c = [0,1,0,0]
        g = [0,0,1,0], t = [0,0,0,1]
    """
    alphabet = 'ACGT'
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
    return np.asarray(onehot_encoded)

if __name__ == "__main__":
    dataset = EnhancersDataset(100)

    print(dataset[99])
    print("Done")