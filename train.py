from typing import Optional

import torchvision.transforms as transforms
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.networks import Generator, Discriminator
from model.utils import *
from process_data import create_dir, Dataset


def train_loop(
        generator: Generator,
        discriminator: Discriminator,
        dataloader: torch.utils.data.DataLoader,
        gen_optimizer: Optimizer,
        disc_optimizer: Optimizer,
        z_dim: int,
        save_step: int = 500,
        save_dir: str = "weights",
        crit_repeats: int = 1,
        n_epochs: int = 1000,
        c_lambda: float = 10,  # need to be changed
        device: str = "cuda",
        display_step: Optional[int] = None,
        lazy_gradient_penalty: int = 4,
) -> None:
    cur_step = 0

    generator_losses = []
    critic_losses = []

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

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                # gradient = gradient_of(discriminator, real, fake.detach(), epsilon)

                disc_loss = discriminator_loss(disc_real_pred, disc_fake_pred)
                if cur_step % lazy_gradient_penalty == 0:
                    gp = gradient_penalty(real, disc_real_pred)
                    disc_loss += gp * c_lambda

                # Keep track of the average critic loss in this batch
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

            # Update the weights
            gen_optimizer.step()
            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            if cur_step % save_step == 0 and cur_step > 0:
                torch.save(generator.state_dict(), f"{save_dir}/generator_{cur_step}.pth")
                torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_{cur_step}.pth")

            # Visualization code
            if display_step:
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
                        label="Discriminator Loss"
                    )
                    plt.legend()
                    plt.show()

            cur_step += 1


def train(epochs: int, dry_run: bool = False, kaggle_mode: bool = False):
    z_dim = 128
    w_dim = 256
    n_mapping_layers = 5
    image_resolution = 32

    display_step = 500
    batch_size = 8

    synthesis_lr = 2e-5
    synthesis_betas = (0.5, 0.9)

    mapping_lr = 2e-5
    mapping_betas = (0.5, 0.9)

    discriminator_lr = 2e-5
    discriminator_betas = (0.5, 0.9)

    save_step = 500
    save_path = "weights/"
    create_dir(save_path)

    c_lambda = 10
    crit_repeats = 5
    n_epochs = epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = "data/landscapes" if not kaggle_mode else "../input/landscapes/landscapes"  # Need to be changed
    dataset = Dataset(dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    gen = Generator(
        z_dim=z_dim,
        w_dim=w_dim,
        image_resolution=image_resolution,
        n_mapping_layers=n_mapping_layers
    ).to(device)
    disc = Discriminator(resolution=image_resolution).to(device)

    gen_opt = Adam([
        {'params': gen.synthesis.parameters(), 'lr': synthesis_lr, 'betas': synthesis_betas},
        {'params': gen.mapping.parameters(), 'lr': mapping_lr, 'betas': mapping_betas}
    ])
    disc_opt = Adam(disc.parameters(), lr=discriminator_lr, betas=discriminator_betas)

    if dry_run:
        noise = get_noise((batch_size, z_dim), device=device)
        fake = gen(noise)
        show_tensor_images(fake, num_images=batch_size)
        return

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
        save_dir=save_path,
        display_step=display_step
    )


def main() -> None:
    epochs = 100_000
    torch.cuda.set_per_process_memory_fraction(0.5)
    train(epochs, dry_run=False)


if __name__ == '__main__':
    main()
