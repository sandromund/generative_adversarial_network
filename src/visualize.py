import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from const import ONE_HOT_ENCODING


def plot_generated_sample(sample):
    if ONE_HOT_ENCODING:
        track_x = np.argmax(sample.detach().cpu().numpy()[0], axis=-1)
        plt.imshow(track_x, interpolation='none', cmap=cm.binary)
        plt.show()
    else:
        plt.imshow(sample.cpu().numpy()[0], interpolation='none')
        plt.show()
