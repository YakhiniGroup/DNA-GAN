import os
import errno

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from utils.sequence_utils import *


class ChrDataset(Dataset):
    """Chromosome dataset, obtained from http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/

    attributes:
           raw_filename (str): The name of the raw .fa file contatining the dataset.
           raw_folder (str): The name of the folder containing the raw file, usually will be raw
           processed_folder (str): The name of the folder containing processed file, after we turn it to .pt serialized
           training_file (str): Name of training serialized processed file.
           test_file (str): Name of test serialized processed file.
    """

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root_folder, sequence_size, raw_file="", train=True, download=False, sample=False):
        r"""Chromosome1 dataset, obtained from http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/

            Args:
                root_folder (string): Directory with all data.
                sequence_size (int): size of splice to split chromosome to.
                download (bool): Should the file be downloaded from internet or not
                sample (bool): Should we sample from dataset or simply split to equal parts
        """
        self.root_folder = root_folder
        self.sequence_size = sequence_size
        self.train = train
        self.sample = sample
        if sample:
            self.raw_folder = os.path.join(self.raw_folder, "sampled")
        if raw_file:
            self.raw_filename = raw_file

        if download:
            self.download()

        # TODO - Maybe add check here instead of in function call.
        self.process(sequence_size)

        # For now we work without labels, simply ignoring them. This is also part of the logic of process.
        if self.train:
            self.data = torch.load(
                os.path.join(self.root_folder, self.processed_folder, self.training_file))
        else:
            self.data = torch.load(
                os.path.join(self.root_folder, self.processed_folder, self.test_file))

    def __len__(self):
        """Get length of data"""
        return len(self.data)

    def __getitem__(self, index):
        """Get instance of data at index"""
        if self.sample:
            return self.data[index:index + self.sequence_size]
        else:
            return self.data[index]

    def download(self):
        """Download the chr1 data if it doesn't exist in raw folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists_raw():
            print("Raw file already exists, skippig download.")
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root_folder, self.raw_folder))
            os.makedirs(os.path.join(self.root_folder, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root_folder, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

    def _check_exists_raw(self):
        return os.path.exists(os.path.join(self.root_folder, self.raw_folder, self.raw_filename))

    def _check_exists_processed(self):
        return os.path.exists(os.path.join(self.root_folder, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root_folder, self.processed_folder, self.test_file))

    def process(self, sequence_size: int):
        r"""Process the raw data of a .fa file of a single very large sequence.
        Most importantly this splits the file into training and test set

        Args:
            sequence_size (int): The size of sequence to split the data set to.
        """

        if self._check_exists_processed():
            return

        f = open(os.path.join(self.root_folder, self.raw_folder, self.raw_filename), 'r')
        # skip the first line
        f.readline()
        # We can change this to create the tensor with a loop, but it will still be O(n)
        # probably, and for now I progress
        full_sequence = f.read().replace('\n', '').lower().replace('n', '')
        full_sequence_length = len(full_sequence)

        if not self.sample:
            subsequences_list = [full_sequence[i:i+sequence_size] for i in range(0, full_sequence_length, sequence_size)]

            # If we have sequence of bad length, for now we dispose.
            if len(subsequences_list[-1]) != sequence_size:
                subsequences_list.pop()

            # Here randomness in introduced! important if I want to restore, I must control this seed as well.
            train_set, test_set = train_test_split(subsequences_list, test_size=0.2, random_state=1234)

            one_hot_train_set = torch.FloatTensor(list(map(sequence_to_one_hot, train_set)))
            one_hot_test_set = torch.FloatTensor(list(map(sequence_to_one_hot, test_set)))

            with open(os.path.join(self.root_folder, self.processed_folder, self.training_file), 'wb') as f:
                torch.save(one_hot_train_set, f)

            with open(os.path.join(self.root_folder, self.processed_folder, self.test_file), 'wb') as f:
                torch.save(one_hot_test_set, f)

        else:
            # Here we don't shuffle, this will completly mess up the dataset.
            train_set, test_set = train_test_split(full_sequence, test_size=0.2,)
            with open(os.path.join(self.root_folder, self.processed_folder, self.training_file), 'wb') as f:
                torch.save(train_set, f)

            with open(os.path.join(self.root_folder, self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)


class Chr1Dataset(ChrDataset):
    r"""Chromosome1 dataset, obtained from http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/"""
    urls = [
        'http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz',
    ]

    raw_filename = 'chr1.fa'


class Chr1DevDataset(ChrDataset):
    r"""Chromosome1 dev dataset, obtained from http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/"""
    urls = [
        'http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1_GL383518v1_alt.fa.gz',
    ]

    raw_filename = 'chr1_GL383518v1_alt.fa'

