import mlflow.pytorch
import torch
from torch import nn
from tqdm import tqdm

from data import get_demo_data_loader
from model import Generator, Discriminator


def train_models(batch_size, lr, num_epochs):
    mlflow.set_experiment("Generative Adversarial Network")
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        mlflow.set_tag("GAN", "DL")

        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('num_epochs', num_epochs)

        discriminator = Discriminator()
        generator = Generator()
        train_loader = get_demo_data_loader(batch_size=batch_size)
        loss_function = nn.BCELoss()

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

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

                mlflow.log_metric("loss_discriminator", loss_discriminator, step=epoch)
                mlflow.log_metric("loss_generator", loss_generator, step=epoch)

        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")
