import datetime
import os.path

import click
from matplotlib import pyplot as plt
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from constants import *
from dataset import *
from distributed import wrap_to_ddp, setup_ddp, destroy_ddp
from model.discriminator import Discriminator
from model.generator import Generator
from util.evaluation import *
from util.files import *
from util.visualization import *


def train_loop(
        generator: Generator,
        discriminator: Discriminator,
        dataloader: torch.utils.data.DataLoader,
        gen_optimizer: Optimizer,
        disc_optimizer: Optimizer,
        z_dim: int,
        save_step: int = 506,
        crit_repeats: int = 1,
        n_epochs: int = 1000,
        c_lambda: float = 10,  # need to be changed
        device: int | str | torch.device = 0,  # shortcut for cuda:0
        display_step: Optional[int] = None,
        lazy_gradient_penalty: int = 4,
        save_graphics: bool = False,
        mode: str = LOCAL
) -> None:
    cur_step = 0

    generator_losses = []
    critic_losses = []

    if not os.path.exists(LOSS_PATH):
        dump_json(LOSS_PATH, {"generator": [], "discriminator": []})

    losses = fetch_json(LOSS_PATH)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for real in tqdm(dataloader):
            cur_batch_size = len(real)

            real = real.to(device)
            fake = None

            if cur_step % lazy_gradient_penalty == 0:
                real.requires_grad_()

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                disc_optimizer.zero_grad()
                fake_noise = get_noise((cur_batch_size, z_dim), device=device)
                fake = generator(fake_noise)
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(real)

                disc_loss = discriminator_loss(disc_real_pred, disc_fake_pred)
                if cur_step % lazy_gradient_penalty == 0:
                    gp = gradient_penalty(real, disc_real_pred)
                    disc_loss += gp * c_lambda

                mean_iteration_critic_loss += disc_loss.item() / crit_repeats
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
            critic_losses += [mean_iteration_critic_loss]

            gen_optimizer.zero_grad()
            fake_noise_2 = get_noise((cur_batch_size, z_dim), device=device)
            fake_2 = generator(fake_noise_2)
            disc_fake_pred = discriminator(fake_2)

            gen_loss = generator_loss(disc_fake_pred)
            gen_loss.backward()

            gen_optimizer.step()
            generator_losses += [gen_loss.item()]

            if cur_step % save_step == 0 and cur_step > 0:
                if mode == LOCAL:
                    torch.save(generator.state_dict(), f"{WEIGHTS_PATH}/generator_{epoch}.pth")
                    torch.save(discriminator.state_dict(), f"{WEIGHTS_PATH}/discriminator_{epoch}.pth")
                else:
                    torch.save(generator.module.state_dict(), f"{WEIGHTS_PATH}/generator_{epoch}.pth")
                    torch.save(discriminator.module.state_dict(), f"{WEIGHTS_PATH}/discriminator_{epoch}.pth")

            if display_step:
                if cur_step % display_step == 0 and cur_step > 0:
                    visualize(generator_losses, critic_losses, fake, real, display_step, epoch, cur_step, save_graphics)
            cur_step += 1

        losses.get("generator").append(generator_losses)
        losses.get("discriminator").append(critic_losses)
        dump_json(LOSS_PATH, losses)

def visualize(
        generator_losses: list,
        critic_losses: list,
        fake: torch.Tensor,
        real: torch.Tensor,
        n_last: int, epoch: int,
        cur_step: int,
        save: bool = False
) -> None:
    gen_mean = sum(generator_losses[-n_last:]) / n_last
    crit_mean = sum(critic_losses[-n_last:]) / n_last
    print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
    if save:
        show_tensor_images(fake, save_path=f"{SAMPLE_PATH}/fake_sample_{epoch}.png")
        show_tensor_images(real, save_path=f"{SAMPLE_PATH}/real_sample_{epoch}.png")
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
        label="Discriminator Loss"
    )
    plt.legend()
    plt.show()
    if save:
        plt.savefig(f"{PLOT_PATH}/plot_{epoch}.png")
    plt.close()


@click.command()
@click.option('--num_epochs', default=100_000, help='Number of epochs to train the model for. '
                                                    '100_000 is the default.')
@click.option('--mode', default='local', help='Mode to run the training in.'
                                              '`local` is the default.'
                                              '`local` will run the training locally. '
                                              '`mgpu` will run the training on multiple GPUs. '
                                              '`mnode` will run the training on multiple nodes.')
@click.option('--resolution', default=32, help='Resolution of the images to train on. 32 is the default.')
@click.option('--display-step', default=506, help='Number of steps to display the images for. The 506 is the default.'
                                                  'If none is given, it will not display the images.')
@click.option('--save-step', default=506, help='Number of steps to save the images for. 506 is the default.')
@click.option('--batch-size', default=8, help='Batch size to use for training. 8 is the default.')
@click.option('--save-graphics', default=False, help='Whether to save the graphics. False is the default.')
@click.option('--rank', default=0, help='Rank of the current node. 0 is the default server node.')
def train(num_epochs: int,
          mode: str,
          rank: int,
          resolution: int,
          display_step: int,
          save_step: int,
          batch_size: int,
          save_graphics: bool) -> None:
    z_dim = 128
    w_dim = 256
    n_mapping_layers = 5

    synthesis_lr = 2e-5
    synthesis_betas = (0.5, 0.9)

    mapping_lr = 2e-5
    mapping_betas = (0.5, 0.9)

    discriminator_lr = 2e-5
    discriminator_betas = (0.5, 0.9)

    c_lambda = 10
    crit_repeats = 5
    n_epochs = num_epochs

    formatted_date = datetime.datetime.now().strftime("%d%m%y")
    if not os.path.exists(WEIGHTS_PATH):
        create_dir_or_ignore(WEIGHTS_PATH)
    dir_version = f"v{len(filter_by_dirname(WEIGHTS_PATH, formatted_date)) + 1}"

    save_path = os.path.join(WEIGHTS_PATH, formatted_date, dir_version)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} already exists.")

    if save_graphics:
        to_be_checked = ["plots", "samples"]
        for folder in to_be_checked:
            if not os.path.exists(folder):
                plot_dir = os.path.join(folder, formatted_date, dir_version)
                create_dir_or_ignore(plot_dir)
            else:
                print(f"Directory {folder} already exists.")

    dataset = Dataset(DATA_PATH, crop_size=resolution)

    gen = Generator(z_dim=z_dim, w_dim=w_dim, image_resolution=resolution, n_mapping_layers=n_mapping_layers)
    disc = Discriminator(resolution=resolution, n_features=resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if mode == LOCAL:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = gen.to(device)
        disc = disc.to(device)
        gen_opt = Adam([
            {'params': gen.synthesis.parameters(), 'lr': synthesis_lr, 'betas': synthesis_betas},
            {'params': gen.mapping.parameters(), 'lr': mapping_lr, 'betas': mapping_betas}
        ])
        disc_opt = Adam(disc.parameters(), lr=discriminator_lr, betas=discriminator_betas)
        train_loop(
            generator=gen, discriminator=disc,
            dataloader=dataloader,
            gen_optimizer=gen_opt,
            disc_optimizer=disc_opt,
            z_dim=z_dim,
            crit_repeats=crit_repeats,
            n_epochs=n_epochs,
            c_lambda=c_lambda,
            device=device,
            save_step=save_step,
            display_step=display_step, )
    elif mode in (MULTI_GPU, MULTI_NODE):
        world_size = torch.cuda.device_count()
        setup_ddp(rank, world_size)
        device = f"cuda:{rank}"
        dataloader.sampler = DistributedSampler(dataset, rank=rank)
        gen, disc = wrap_to_ddp([gen, disc], rank)
        gen_opt = Adam([
            {'params': gen.synthesis.parameters(), 'lr': synthesis_lr, 'betas': synthesis_betas},
            {'params': gen.mapping.parameters(), 'lr': mapping_lr, 'betas': mapping_betas}
        ])
        disc_opt = Adam(disc.parameters(), lr=discriminator_lr, betas=discriminator_betas)
        mp.spawn(fn=train_loop, args=(gen, disc, dataloader, gen_opt, disc_opt, z_dim, crit_repeats,
                                      n_epochs, c_lambda, device, save_step, display_step, mode)
                 , nprocs=world_size)
        destroy_ddp()
    else:
        raise ValueError(f"Mode {mode} is not supported.")

    if mode in (MULTI_GPU, MULTI_NODE):
        pass


if __name__ == '__main__':
    train()
