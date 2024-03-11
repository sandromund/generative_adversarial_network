import matplotlib.pyplot as plt
import torch

from model import Generator


def generate_sample(model_path):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path))
    print(generator.eval())
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show()
