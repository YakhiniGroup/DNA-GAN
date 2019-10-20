import torch
from torch.utils.data import DataLoader

from models.WGANGP import WGANGP

from utils.tensor_utils import *
from utils.sequence_utils import *
from utils.language_utils import KmerLanguageModel

import tensorboardX

from abc import ABC, abstractmethod

from collections import defaultdict
import itertools


class WGANGPEvaluator(ABC):
    r"""base class for all evaluators

    An evaluator is an object that receives a gan model, a summary writer and a dataset.
    It evaluates data that the model generates compared to desired properties to be found in the data.
    For example, an evaluator can compare the KL divergence between distributions of bases in the generated data, to
    distribution of bases in the given dataset.

    The evaluator receives a SummaryWriter object from a trainer, so that it can be used for evaluating the model
    even during training. This helps encapsulates the logic of training from a user, which in order to preform
    some experiment, needs only to implement his own evaluator, and pass it to the wgangp trainer object.

    """
    def __init__(self, evaluated_wgangp: WGANGP,
                 writer: tensorboardX.SummaryWriter,
                 data_loader: DataLoader,
                 **kwargs) -> None:
        r"""

        Args:
            evaluated_wgangp: A wgangp to evaluate

        """
        self.evaluated_wgangp = evaluated_wgangp
        self.writer = writer
        self.data_loader = data_loader

        self.num_samples = kwargs.pop('num_samples', 1280)

    def generate_batch(self, batch_size):
        # Sample a batch of noise to generate samples
        batch_of_noise = to_cuda(torch.randn(batch_size, self.evaluated_wgangp.generator.latent_dim))
        batch_of_generated = self.evaluated_wgangp.generator(batch_of_noise).detach()
        return batch_of_generated

    @abstractmethod
    def evaluate(self, iteration=0) -> None:
        """Evaluate the performance of the given wgan, performance of evaluated_model, at the given iteration
        """
        self.evaluated_wgangp.eval()


class KmerEvaluator(WGANGPEvaluator):
    def __init__(self, evaluated_wgangp: WGANGP,
                 writer: tensorboardX.SummaryWriter,
                 data_loader: DataLoader,
                 **kwargs
                 ) -> None:
        super().__init__(evaluated_wgangp, writer, data_loader)

        self.ks = kwargs.pop('ks', [])
        self.num_validation_batches = kwargs.pop('num_validation_batches', 10)
        self.final_kmer_language_model = {}
        self.js_with_real_history = defaultdict(list)
        self.kmer_proportions_history = defaultdict(list)

        # These are immutable static values that should not change during training
        self.optimal_js_values = {}
        self.js_random_to_real = {}
        self.real_kmer_model = {}

        # TODO - Change this name it's bad
        self._init()

    def _init(self):
        r"""This method sets up the values required for the kmer evaluator to make calculations and comparisons.
        For a given k, when  preforming the kmer evaluation, we need to have a kmer model of the real data in order
        to take a metric between it and the generated sequences.

        Furthermore, as we wish to have some sort of baseline to compare to, we initialize a "null model", which is
        a comparision between a random generated sequence and the real data
        """
        batches = []
        rest_of_batches = []
        for num_batch, batch in enumerate(self.data_loader):
            if num_batch < self.num_validation_batches:
                batches.append(batch)
            else:
                rest_of_batches.append(batch)

        # TODO - might be memory intensive, we could feed it batch by batch and calculate js, then average.
        # not sure this is required.
        first = torch.cat(batches)
        rest = torch.cat(rest_of_batches)
        full = torch.cat((first, rest))

        random_sequence = generate_random_sequence(full.size()[0] * full.size()[1])

        for k in self.ks:
            # Calculate the optimal kmer value to achieve, by comparing batches of dataset to itself.
            val = self.create_kmer_language_model(k, first)
            true = self.create_kmer_language_model(k, rest)
            self.optimal_js_values[k] = true.js_with(val)

            # Store a language model for the full dataset for given k
            current_kmer_model = self.create_kmer_language_model(k, full)
            self.real_kmer_model[k] = current_kmer_model

            random_kmer_model = KmerLanguageModel(k, random_sequence)
            self.js_random_to_real[k] = current_kmer_model.js_with(random_kmer_model)

    @staticmethod
    def create_kmer_language_model(k, batch):

        one_hot = batch_to_one_hot(batch)

        # TODO - worry if it's on cpu or not?
        one_hot_flattened = one_hot.reshape(-1, 4)

        batch_as_sequence = vector_to_seq(one_hot_flattened)

        return KmerLanguageModel(k, batch_as_sequence)

    def evaluate(self, iteration=0):
        super().evaluate()
        for k in self.ks:
            # Generate batces and create kmer language model for them
            generated_batch = self.generate_batch(self.num_samples)
            current_kmer_model = self.create_kmer_language_model(k, generated_batch)

            # Store current language model, not sure this is neccesary, maybe we can store history
            self.final_kmer_language_model[k] = current_kmer_model

            # TODO - Store counts history, this is a subset of the data found in language model,
            #  maybe we should store it.
            keys = list(map(lambda x: ''.join(x), itertools.permutations('acgt', k)))

            # This initiate it such that all kmers not having appeared have proportion zero
            # we don't want a default dict here since this object should not be mutable.
            current_counts = dict.fromkeys(keys, 0)

            for kmer in self.final_kmer_language_model[k].kmers():
                count_of_kmer = self.final_kmer_language_model[k]._kmer_counts[kmer]
                current_counts[kmer] = count_of_kmer

            self.kmer_proportions_history[k].append(current_counts)

            current_js = current_kmer_model.js_with(self.real_kmer_model[k])
            self.js_with_real_history[k].append(current_js)

            if iteration > 0:
                self.writer.add_scalars('evaluation/js_with_real_%d' % k,
                                        {'val_js' : current_js,
                                         'optimal_js' : self.optimal_js_values[k],
                                         'random_js' : self.js_random_to_real[k]},
                                        iteration)


class WasserstainDistanceEvaluator(WGANGPEvaluator):
    def __init__(self, evaluated_wgangp: WGANGP,
                 writer: tensorboardX.SummaryWriter = None,
                 **kwargs) -> None:
        super(WasserstainDistanceEvaluator, self).__init__(evaluated_wgangp, writer)

    @staticmethod
    def w_dist(wgangp, batch):
        # Sample a batch of noise to generate samples
        batch_of_noise = to_cuda(torch.randn(batch.size()[0], wgangp.generator.latent_dim))
        batch_of_generated = wgangp.generator(batch_of_noise).detach()
        D_generated = wgangp.discriminator(batch_of_generated)
        D_generated = D_generated.mean()

        D_real = wgangp.discriminator(batch)
        D_real = D_real.mean()

        w = D_generated - D_real
        return w.item()

    def diff_of_w_dist(self, validation_batch: torch.Tensor):
        return self.w_dist(self.random_wgangp, validation_batch) - self.w_dist(self.evaluated_wgangp, validation_batch)

    def evaluate(self):
        super().evaluate()