from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import NLLLoss

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model.components import Generator, Critic
from utils import *


# def run_epoch(
#         gen: Generator,
#         disc: Discriminator,
#         loader: DataLoader,
#         gen_opt: Adam,
#         disc_opt: Adam,
#         z_dim: int = 512,
#         disc_repeats: int = 1,
#         c_lambda: float = 10,
#         device="cuda"
# ) -> tuple:
#     for real, _ in tqdm(loader, leave=True):
#         batch_size = len(real)
#         real = real.to(device)
#
#         mean_iteration_disc_loss = 0
#         for _ in range(disc_repeats):
#             disc_opt.zero_grad()
#
#             fake_noise = get_noise((batch_size, z_dim), device=device)
#             fake = gen(fake_noise)
#             disc_fake_pred = disc(fake.detach())
#             disc_real_pred = disc(real)
#
#             epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
#             gradient = get_gradient(disc, real, fake.detach(), epsilon)
#             gp = gradient_penalty(gradient)
#             crit_loss = critic_loss(
#                 disc_fake_pred, disc_real_pred, gp, c_lambda
#             )
#
#             # Keep track of the average critic loss in this batch
#             mean_iteration_disc_loss += crit_loss.item() / disc_repeats
#             crit_loss.backward(retain_graph=True)
#             disc_opt.step()
#
#         disc_losses += [mean_iteration_disc_loss]
#         gen_opt.zero_grad()
#
#         fake_noise_2 = get_noise((batch_size, z_dim), device=device)
#         fake_2 = gen(fake_noise_2)
#         disc_fake_pred_2 = disc(fake_2)
#         gen_loss = generator_loss(disc_fake_pred_2)
#
#         gen_loss.backward()
#         gen_opt.step()
#
#         return np.mean(disc_losses), gen_loss.item()


def train_loop(
        generator: Generator,
        critic: Critic,
        dataloader: torch.utils.data.DataLoader,
        gen_optimizer: Optimizer,
        crit_optimizer: Optimizer,
        z_dim: int,
        crit_repeats: int = 1,
        n_epochs: int = 1000,
        c_lambda: float = 10,  # need to be changed
        device: str = "cuda",
        display_step=None
):
    cur_step = 0

    generator_losses = []
    critic_losses = []

    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)

            real = real.to(device)
            fake = None

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                crit_optimizer.zero_grad()
                fake_noise = get_noise((cur_batch_size, z_dim), device=device)
                fake = generator(fake_noise)
                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = gradient_of(critic, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            critic_losses += [mean_iteration_critic_loss]

            gen_optimizer.zero_grad()
            fake_noise_2 = get_noise((cur_batch_size, z_dim), device=device)
            fake_2 = generator(fake_noise_2)
            crit_fake_pred = critic(fake_2)

            gen_loss = generator_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_optimizer.step()
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            # Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()

            cur_step += 1
