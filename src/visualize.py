import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_one_hot_encoded_track(track_x):
    track_x = np.argmax(track_x.cpu().numpy(), axis=-1)
    plt.imshow(track_x, interpolation='none', cmap=cm.binary)
    plt.show()
