LATENT_SPACE_SAMPLE = 100
ONE_HOT_ENCODING = True
if ONE_HOT_ENCODING:
    DATA_DIMENSION = 36 * 36 * 4
else:
    DATA_DIMENSION = 36 * 36
BETAS = (0.5, 0.999)  # decay of first and second  order momentum of gradient
SEED = 42
DISCRIMINATOR_DELAY = 5
