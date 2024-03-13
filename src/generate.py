import matplotlib.pyplot as plt
import torch

from const import LATENT_SPACE_SAMPLE
import mlflow


def generate_sample_mlflow(model_uri):
    generator = mlflow.pytorch.load_model(model_uri)
    latent_space_samples = torch.randn(1, LATENT_SPACE_SAMPLE)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    generated_samples = generated_samples.cpu().numpy()[0]
    plt.imshow(generated_samples, interpolation='none')
    plt.show()
