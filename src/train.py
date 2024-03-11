import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from data import get_demo_data_loader
from model import Generator, Discriminator


def train_models(batch_size, model_save_path, lr, num_epochs, plot_training):
    discriminator = Discriminator()
    generator = Generator()
    train_loader = get_demo_data_loader(batch_size=batch_size)
    loss_function = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    loss_discriminator_list = []
    loss_generator_list = []

    for epoch in tqdm(range(num_epochs)):
        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2))

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            loss_discriminator_list.append(float(loss_discriminator))
            loss_generator_list.append(float(loss_generator))

    if plot_training:
        plt.plot(loss_discriminator_list, label='loss_discriminator')
        plt.plot(loss_generator_list, label='loss_generator')
        plt.legend()
        plt.show()

    torch.save(generator.state_dict(), model_save_path)
