import mlflow.pytorch
import torch
from torch import nn
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from tqdm import tqdm

import const as c
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
        mlflow.log_param("LATENT_SPACE_SAMPLE".lower(), c.LATENT_SPACE_SAMPLE)
        mlflow.log_param("ONE_HOT_ENCODING".lower(), c.ONE_HOT_ENCODING)
        mlflow.log_param("BETAS".lower(), c.BETAS)
        mlflow.log_param("SEED".lower(), c.SEED)
        mlflow.log_param("WEIGHT_BCE_LOSS".lower(), c.WEIGHT_BCE_LOSS)
        mlflow.log_param("WEIGHT_CUSTOM_LOSS".lower(), c.WEIGHT_CUSTOM_LOSS)
        mlflow.log_param("DISCRIMINATOR_ROUND_OUTPUT".lower(), c.DISCRIMINATOR_ROUND_OUTPUT)

        discriminator = Discriminator().to(device)
        generator = Generator().to(device)
        train_loader = get_data_loader(data_path=data_path, batch_size=batch_size)
        loss_function = nn.BCELoss().to(device)  # binary cross entropy loss function
        f1_bin = BinaryF1Score().to(device)
        recall = BinaryRecall().to(device)
        precision = BinaryPrecision().to(device)

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=c.BETAS)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=c.BETAS)

        loss_generator = 0
        for epoch in tqdm(range(num_epochs), "train_models"):
            for n, (real_samples, _) in enumerate(train_loader):
                real_samples = real_samples.to(device=device)
                # Data for training the discriminator
                real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
                latent_space_samples = torch.randn((batch_size, c.LATENT_SPACE_SAMPLE)).to(device=device)
                generated_samples = generator(latent_space_samples).to(device=device)
                generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
                all_samples = torch.cat((real_samples, generated_samples)).to(device=device)
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(device=device)

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples).to(device=device)

                loss_discriminator = loss_function(output_discriminator, all_samples_labels)
                if loss_discriminator > loss_generator:
                    loss_discriminator.backward()
                    optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, c.LATENT_SPACE_SAMPLE)).to(device=device)

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples).to(device=device)
                output_discriminator_generated = discriminator(generated_samples).to(device=device)

                bce_loss_loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
                # custom_loss_generator = torch.mean(torch.abs(output_discriminator_generated -
                #                                             torch.zeros_like(output_discriminator_generated)))
                # custom_loss_generator = torch.abs(output_discriminator_generated.sum() - real_samples.sum())
                non_zero_generated = torch.count_nonzero(output_discriminator_generated)
                non_zero_real_samples_ = torch.count_nonzero(real_samples)
                custom_loss_generator = torch.abs(non_zero_generated - non_zero_real_samples_) / 100
                loss_generator = c.WEIGHT_BCE_LOSS * bce_loss_loss_generator + c.WEIGHT_CUSTOM_LOSS * custom_loss_generator
                if loss_generator > loss_discriminator:
                    loss_generator.backward()
                    optimizer_generator.step()

            mlflow.log_metric("output_discriminator_generated_sum", output_discriminator_generated.sum(), step=epoch)
            mlflow.log_metric("output_discriminator_sum", output_discriminator.sum(), step=epoch)

            mlflow.log_metric("f1_discriminator",
                              f1_bin(output_discriminator, all_samples_labels),
                              step=epoch)
            mlflow.log_metric("f1_generator",
                              f1_bin(output_discriminator_generated, real_samples_labels),
                              step=epoch)

            mlflow.log_metric("precision_discriminator",
                              precision(output_discriminator, all_samples_labels),
                              step=epoch)
            mlflow.log_metric("precision_generator",
                              precision(output_discriminator_generated, real_samples_labels), step=epoch)
            mlflow.log_metric("recall_discriminator",
                              recall(output_discriminator, all_samples_labels), step=epoch)
            mlflow.log_metric("recall_generator",
                              recall(output_discriminator_generated, real_samples_labels), step=epoch)
            mlflow.log_metric("loss_discriminator", loss_discriminator, step=epoch)
            mlflow.log_metric("loss_generator", loss_generator, step=epoch)
            mlflow.log_metric("loss_bce_generator", bce_loss_loss_generator, step=epoch)
            mlflow.log_metric("loss_custom_generator", custom_loss_generator, step=epoch)
            mlflow.log_metric("non_zero_generated", non_zero_generated, step=epoch)
            mlflow.log_metric("non_zero_real_samples", non_zero_real_samples_, step=epoch)

        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")
