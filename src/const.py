LATENT_SPACE_SAMPLE = 64
ONE_HOT_ENCODING = True
if ONE_HOT_ENCODING:
    DATA_DIMENSION = 36 * 36 * 4
else:
    DATA_DIMENSION = 36 * 36
BETAS = (0.5, 0.999)  # decay of first and second  order momentum of gradient
SEED = 42
WEIGHT_BCE_LOSS = 0.9  # Weight for the BCELoss
WEIGHT_CUSTOM_LOSS = 1 - WEIGHT_BCE_LOSS  # Weight for the custom loss function
DISCRIMINATOR_ROUND_OUTPUT = True
