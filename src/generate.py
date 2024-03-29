import mlflow
import torch

from const import LATENT_SPACE_SAMPLE
from visualize import plot_generated_sample


def generate_sample_mlflow(model_uri):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = mlflow.pytorch.load_model(model_uri)
    latent_space_samples = torch.randn(1, LATENT_SPACE_SAMPLE).to(device=device)
    generated_sample = generator(latent_space_samples)
    plot_generated_sample(generated_sample)
