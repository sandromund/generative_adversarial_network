import logging

import mlflow.pytorch
import torch
from torch import nn
from tqdm import tqdm

from const import LATENT_SPACE_SAMPLE, ONE_HOT_ENCODING, BETAS
from data import get_data_loader
from model import Generator, Discriminator


def train_models(data_path, batch_size, lr, num_epochs):
    mlflow.set_experiment("Generative Adversarial Network")
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.set_tag("device", device)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param("LATENT_SPACE_SAMPLE".lower(), LATENT_SPACE_SAMPLE)
        mlflow.log_param("ONE_HOT_ENCODING".lower(), ONE_HOT_ENCODING)
        mlflow.log_param("BETAS".lower(), BETAS)

        discriminator = Discriminator().to(device)
        generator = Generator().to(device)
        train_loader = get_data_loader(data_path=data_path, batch_size=batch_size)
        loss_function = nn.BCELoss().to(device)  # binary cross entropy loss function

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=BETAS)

        for epoch in tqdm(range(num_epochs), "train_models"):
            for n, (real_samples, _) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
                latent_space_samples = torch.randn((batch_size, LATENT_SPACE_SAMPLE)).to(device=device)
                generated_samples = generator(latent_space_samples).to(device=device)
                generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
                all_samples = torch.cat((real_samples.to(device=device), generated_samples)).to(device=device)
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(device=device)

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples).to(device=device)

                loss_discriminator = loss_function(output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, LATENT_SPACE_SAMPLE)).to(device=device)

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples).to(device=device)
                output_discriminator_generated = discriminator(generated_samples).to(device=device)
                loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
                loss_generator.backward()
                optimizer_generator.step()

                mlflow.log_metric("loss_discriminator", loss_discriminator, step=epoch)
                mlflow.log_metric("loss_generator", loss_generator, step=epoch)
                mlflow.log_metric("generated_samples_sum", generated_samples.sum() / batch_size, step=epoch)

        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")
