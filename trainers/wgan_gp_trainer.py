import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils.sequence_utils import *
from utils.tensor_utils import *
from evaluators.wgan_gp_evaluator import KmerEvaluator
from models.WGANGP import WGANGP


class WGANGPTrainer:
    r"""A class to train a wgangp network.
    Write a very very long explanation documenting all of this
    """
    def __init__(self, wgan: WGANGP,
                 train_data_loader: DataLoader,
                 validation_data_loader: DataLoader,
                 **kwargs) -> None:
        r"""Construct a new trainer instance

        Args:
            model (WGANGP object): A WGANGP, holding a discriminator and generator.
            discriminator_iterations (int): The number of iterations the discriminator prefroms per generator iter.
            output_folder (str): Location to save models if wanted.
            should_log (bool): Should use tensorboard to log progress of training.
            expirament_name (str): The name of the folder underwhich tensorboard will save the run.
        """
        self.wgan = wgan
        self.train_data_loader = train_data_loader
        self.batch_size = train_data_loader.batch_size
        self.validation_data_loader = validation_data_loader

        # Unpack keyword arguments
        self.discriminator_iterations = kwargs.pop('discriminator_iterations', 5)
        self.LAMBDA = kwargs.pop('LAMBDA', 10)
        self.d_adam_config = kwargs.pop('d_adam_config', {'lr': 1e-4, 'beta1':0.5, 'beta2':0.9})
        self.g_adam_config = kwargs.pop('g_adam_config', {'lr': 1e-4, 'beta1': 0.5, 'beta2': 0.9})
        self.num_epochs = kwargs.pop('num_epochs', 1)
        self.should_train_generator = kwargs.pop('should_train_generator', True)
        self.latent_space = kwargs.pop('latent_space', torch.randn)

        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.output_folder = kwargs.pop('output_folder', 'saved_models')
        self.should_log_tbx = kwargs.pop('should_log_tbx', False)
        if self.should_log_tbx:
            self.experiment_name = kwargs.pop('experiment_name', None)
            self.writer = SummaryWriter(log_dir=os.path.join('runs', self.experiment_name))
        else:
            self.writer = None
        # Default value would result in about 7 displations per epoch.
        self.display_step = kwargs.pop('display_step', 1000)
        self.verbose = kwargs.pop('verbose', False)

        self.evaluate_every = kwargs.pop('evaluate_every', 0)
        self.should_evaluate = self.evaluate_every > 0
        if self.should_evaluate:
            self._setup_evaluator()

        self._setup_optimization()

    def _setup_evaluator(self):
        self.evaluator = KmerEvaluator(evaluated_wgangp=self.wgan,
                                       writer=self.writer,
                                       data_loader=self.validation_data_loader,
                                       ks=[1, 2, 3])

    def _setup_optimization(self):
        """Setup all the parameters tracking history of the model and its
        """
        # Setup all variables storing preformance history of the model
        self.num_trained_epochs = 0
        self.num_trained_iterations = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.D_loss_history = []
        self.G_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Setup the parameters of the adam optimizer to update
        d_lr, d_beta1, d_beta2 = self.d_adam_config['lr'], self.d_adam_config['beta1'], self.d_adam_config['beta2']
        g_lr, g_beta1, g_beta2 = self.g_adam_config['lr'], self.g_adam_config['beta1'], self.g_adam_config['beta2']

        self.D_optimizer = optim.Adam(params=[p for p in self.wgan.discriminator.parameters() if p.requires_grad],
                                      lr=d_lr, betas=(d_beta1, d_beta2))
        self.G_optimizer = optim.Adam(params=[p for p in self.wgan.generator.parameters()if p.requires_grad],
                                      lr=g_lr, betas=(g_beta1, g_beta2))

    # TODO - use weblogo api to draw nice sequences :)
    # it should be implemented in a different class, utils.sequence_visualization_utils
    def generate_sequences(self, batch_size):
        r"""Generate sequences with the current generator to visualize training.
        """

        # This puts all submodules of wgan (discriminator and generator) in eval mode,
        # Removing all regularization
        self.wgan.eval()
        test_batch = to_cuda(self.latent_space(batch_size, self.wgan.generator.latent_dim))
        test_seq = self.wgan.generator(test_batch)

        print("Visualize some random test sequences:")
        print("Test sequence", one_hot_to_seq(test_seq[np.random.randint(batch_size)].max(dim=1)[1]))
        print("Test sequence", one_hot_to_seq(test_seq[np.random.randint(batch_size)].max(dim=1)[1]))
        print("Test sequence", one_hot_to_seq(test_seq[np.random.randint(batch_size)].max(dim=1)[1]))
        print("Test sequence", one_hot_to_seq(test_seq[np.random.randint(batch_size)].max(dim=1)[1]))
        print("Test sequence", one_hot_to_seq(test_seq[np.random.randint(batch_size)].max(dim=1)[1]))

    def calc_gradient_penalty(self, batch_of_real, batch_of_fake):
        r"""Calculate the gradient penalty, which is λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2.

        Args:
            batch_of_real (torch.Tensor): A batch of real samples, from training.
            batch_of_fake (torch.Tensor): A batch of fake samples, generated by generator.
            LAMBDA (int): Hyperparameter controling the strength of gradient penalty.

        Returns:
            λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2
        """

        # Calculate the interpolates x^ from the algorithm.
        epsilon = to_var(torch.rand(batch_of_real.size()[0], 1, 1).expand(batch_of_real.size()))
        interpolates = epsilon * batch_of_real + (1 - epsilon) * batch_of_fake

        # Calculate D(x^)
        disc_interpolates = self.wgan.discriminator(interpolates)

        # The gradient of the output with respect to itself, should be all 1's
        weight = to_cuda(torch.ones(disc_interpolates.size()))

        # The the gradient of D(x^) with respect to x^
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def train_discriminator(self, batch_of_real: torch.Tensor, batch_of_fake: torch.Tensor) -> (float, float, float):
        r"""Train the discriminator on real and fake data, including gradient penalty.

        Args:
            optimizer (optim.Optimizer): An optimizer of the discriminators params, to take gradient steps.
            Will probably be ADAM most of the time.
            batch_of_real (torch.Tensor): A batch of real samples, from training.
            batch_of_fake (torch.Tensor): A batch of fake samples, generated by generator.
            LAMBDA (int): Hyperparameter controlling the strength of gradient penalty.

        Returns:
            Wasserstein loss for discriminator,
            -E[D(x)] - E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]
        """
        # Zero grad from last run or else they accumulate.
        self.wgan.discriminator.zero_grad()

        # train with real
        D_real = self.wgan.discriminator(batch_of_real)
        D_real = D_real.mean()

        # Train with Fake
        D_fake = self.wgan.discriminator(batch_of_fake)
        D_fake = D_fake.mean()

        # train with gradient penalty
        gradient_penalty = self.calc_gradient_penalty(batch_of_real, batch_of_fake)

        D_loss = D_fake - D_real + gradient_penalty
        D_loss.backward()

        # TODO  - add should_train_d, for fine grain control
        self.D_optimizer.step()

        # Return values for logging
        return D_loss.item(), D_real.item(), D_fake.item(), gradient_penalty.item()

    def train_generator(self):
        r"""Train the generator on a batch of fake.

        optimizer (torch.optim): An optimizer of the discriminators params, to take gradient steps.
        Will probably be ADAM most of the time.

        Returns:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Zero grad from last run, or else they accumulate.
        self.wgan.generator.zero_grad()

        # calculate D(G(z))
        noise = to_cuda(self.latent_space(self.batch_size, self.wgan.generator.latent_dim))
        batch_of_generated = self.wgan.generator(noise)
        D_generated = self.wgan.discriminator(batch_of_generated)
        D_generated = D_generated.mean()
        G_loss = -1 * D_generated

        G_loss.backward()

        if self.should_train_generator:
            self.G_optimizer.step()

        return G_loss.item()

    @staticmethod
    def make_batch(iterator: torch.utils.data.DataLoader) -> torch.Tensor:
        """ A function to extract a a batch of real samples from a dataloader.
        Main use of this is to apply next and wrap with an iterator, so we can keep running for several epochs.

        Args:
            iterator: An iterator over a dataset, usually would be a dataloader.

        Returns:
            A batch from the dataset

        """
        batch = next(iterator)
        batch = to_cuda(batch)
        return batch

    def _step(self, train_iter) -> (float, float):
        D_current_loss = []
        for _ in range(self.discriminator_iterations):
            # Prepare real samples for discriminator
            batch_of_real = self.make_batch(train_iter)

            # Prepare generated samples for discriminator
            noise = to_cuda(self.latent_space(self.batch_size, self.wgan.generator.latent_dim))
            batch_of_generated = self.wgan.generator(noise)

            D_loss, D_real, D_fake, grad_pen = self.train_discriminator(batch_of_real, batch_of_generated)
            D_current_loss.append(D_loss)

        # Now it's generators turn to train
        G_loss = self.train_generator()

        # Now log all losses
        self.D_loss_history.append(np.mean(D_current_loss))
        self.G_loss_history.append(G_loss)

        # Monitor with tensorboardX
        # TODO - turn into seperate function that logs nicely.
        if self.should_log_tbx:
            self.writer.add_scalar('Loss/Discriminator', np.mean(D_current_loss), self.num_trained_iterations)
            self.writer.add_scalar('Loss/Generator', G_loss, self.num_trained_iterations)
            self.writer.add_scalar('D_loss_components/D(x)', D_real, self.num_trained_iterations)
            self.writer.add_scalar('D_loss_components/D(G(z))', D_fake, self.num_trained_iterations)
            self.writer.add_scalar('D_loss_components/grad_pen', grad_pen, self.num_trained_iterations)

    def train(self) -> None:
        r""" Run optimization to train the model.

        """
        # Approximate iterations/epoch given discriminator_iterations per epoch
        iters_in_epoch = int(np.ceil(len(self.train_data_loader) / self.discriminator_iterations))

        num_iterations = iters_in_epoch * self.num_epochs
        with tqdm(total=num_iterations) as pbar:
            for epoch in range(1, self.num_epochs + 1):
                # This causes also submodules (discriminator and generator) to be in training mode.
                self.wgan.train()

                # TODO - is this really neccesary or should we just pass train_data_loader?
                train_iterator = iter(self.train_data_loader)
                for _ in range(iters_in_epoch):
                    # TODO - could change this to hold current batch, turning trainer to highly stateful.
                    # not sure this is desired.
                    self._step(train_iterator)

                    # Update batch counter and progress bar
                    self.num_trained_iterations += 1
                    pbar.update(1)

                    # Generate sequences for visualization. This is only good if we some know
                    # quality we want to measure that is not loss function, maybe stragiths of tttt or something.
                    # Anyway, should probably be passed as a funciton.
                    if self.verbose and self.num_trained_iterations % self.display_step == 0:
                        self.generate_sequences(self.batch_size)

                    if self.should_evaluate and self.num_trained_iterations % self.evaluate_every == 0:
                        self.evaluator.evaluate(self.num_trained_iterations)

                # Increment number of train epochs
                self.num_trained_epochs += 1

    def _save_checkpoint(self, state, is_best: bool, filename: str = 'checkpoint.pth') -> None:
        """ Save a checkpoint of training, so that we can resume training from here again

        Args:
            state (dict): dictionary containing everything needed to restore model after run
            is_best (bool): If this is the best model, save it
            filename (str): name of the file under the output folder to save model to
        """
        torch.save(state, os.path.join(self.output_folder, filename))
        if is_best:
            shutil.copyfile(os.path.join(self.output_folder, filename), 'best_model.pth')

    def plot_loss(self):
        # Set style, figure size
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.num_trained_iterations, len(self.D_loss_history)),
                 self.D_loss_history,
                 'r', alpha=0.5)

        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_trained_iterations, len(self.G_loss_history)),
                 self.G_loss_history,
                 'g', alpha=0.7)

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title("WGAN GP Loss over iterations")
        plt.show()

