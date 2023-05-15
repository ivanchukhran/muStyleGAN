from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import NLLLoss

import torchvision.transforms as transforms

from model.components import Generator, Discriminator
from utils import *


def epoch(
        gen: Generator,
        disc: Discriminator,
        loader: DataLoader,
        gen_opt: Adam,
        disc_opt: Adam,
        z_dim: int = 512,
        disc_repeats: int = 1,
        c_lambda: float = 10,
        device="cuda"
) -> tuple:
    for real, _ in tqdm(loader, leave=True):
        batch_size = len(real)
        real = real.to(device)

        mean_iteration_disc_loss = 0
        for _ in range(disc_repeats):
            disc_opt.zero_grad()

            fake_noise = get_noise((batch_size, z_dim), device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_real_pred = disc(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(disc, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = critic_loss(
                disc_fake_pred, disc_real_pred, gp, c_lambda
            )

            # Keep track of the average critic loss in this batch
            mean_iteration_disc_loss += crit_loss.item() / disc_repeats
            crit_loss.backward(retain_graph=True)
            disc_opt.step()

        disc_losses += [mean_iteration_disc_loss]
        gen_opt.zero_grad()

        fake_noise_2 = get_noise((batch_size, z_dim), device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred_2 = disc(fake_2)
        gen_loss = generator_loss(disc_fake_pred_2)

        gen_loss.backward()
        gen_opt.step()

        return np.mean(disc_losses), gen_loss.item()


def train_loop(gen, disc, loader, gen_opt, disc_opt, z_dim, disc_repeats, epochs):
    for epoch in tqdm(range(epochs)):
        disc_loss, gen_loss = epoch(gen, disc, loader, gen_opt, disc_opt, z_dim, disc_repeats)
