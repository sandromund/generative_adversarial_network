import matplotlib.pyplot as plt
import torch

from model import Generator
import mlflow


def generate_sample_torch(model_path):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path))
    print(generator.eval())
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show()


def generate_sample_mlflow(model_uri):
    generator = mlflow.pytorch.load_model(model_uri)
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show()
