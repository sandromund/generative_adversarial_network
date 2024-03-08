from model import Generator
import torch
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = "generator.pt"

generator = Generator()
generator.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(generator.eval())

latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()
