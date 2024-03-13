import json
import math
import os
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class KilterBoardData(Dataset):
    def __init__(self, x, y):
        super(KilterBoardData, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_data_loader(data_path: str, batch_size: int) -> DataLoader:
    pp = Preprocessor()
    x, y = pp.preprocess_data(data_path=data_path)
    pp.info()
    data = KilterBoardData(x=x, y=y)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader


def get_demo_data_loader(batch_size: int, train_data_length=1024) -> DataLoader:
    """
    import matplotlib.pyplot as plt
    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    """
    train_data = torch.zeros((train_data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length)
    train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader


class Preprocessor:
    """
    Class for preprocessing data.

    Attributes:
    -----------
    data_path : str or None
        The path to the directory containing JSON files. Default is None.
    track_counter : int or None
        Counter for the number of tracks processed. Default is None.
    file_counter : int or None
        Counter for the number of files processed. Default is None.
    rejected_tracks_counter : int or None
        Counter for the number of rejected tracks during preprocessing. Default is None.
    x_min : int
        Minimum value for the x-coordinate of placements. Default is 1.
    x_max : int
        Maximum value for the x-coordinate of placements. Default is 35.
    y_min : int
        Minimum value for the y-coordinate of placements. Default is 0.
    y_max : int
        Maximum value for the y-coordinate of placements. Default is 35.
    data_shape : tuple
        Shape of the preprocessed data tensor. Default is (36, 36).
    mapping : dict
        Mapping of placement types to integers. Default mapping is provided.

    Note:
    -----
    This class provides functionality for preprocessing data for further analysis.

    Example:
        import matplotlib.pyplot as plt

        pp = Preprocessor()
        data = pp.preprocess_data(data_path="../data/climbs")
        pp.info()

        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i][1].cpu().numpy())
        plt.show()

    """

    def __init__(self):
        self.n_samples = 0
        self.n_max_samples = 20_000
        self.data_path = None
        self.track_counter = None
        self.file_counter = None
        self.rejected_tracks_counter = None
        self.x_min = 1
        self.x_max = 35
        self.y_min = 0
        self.y_max = 35
        self.data_shape = (self.y_max + 1, self.x_max + 1)
        self.mapping = {'MIDDLE': 1,
                        'FEET-ONLY': 2,
                        'START': 3,
                        'FINISH': 4}

    def preprocess_track(self, track):
        """
        Preprocesses a single track.

        :param track: dict
            A dictionary representing a track containing placements.

        :return: tuple or None
            A tuple containing the preprocessed track ID and track data as a torch tensor.
            Returns None if the track cannot be preprocessed due to missing data or out-of-bounds placements.

        Note:
        -----
        This method preprocesses a single track by mapping placement types to integers based on a predefined mapping.
        It constructs a torch tensor `pp_track_x` representing the track data with placements mapped to integers.
        If any required data is missing or if the placements are out of bounds, the method returns None.
        """
        pp_track_x = torch.zeros(self.data_shape)
        pp_track_id = track.get("uuid")
        for placement in track.get("placements"):
            placement_type = placement.get("type")
            if placement_type is None:
                return None
            placement_type_mapped = self.mapping.get(placement_type)
            if placement_type_mapped is None:
                return None
            x = placement.get("x")
            y = placement.get("y")
            if x < self.x_min or x > self.x_max or self.y_min < 0 or y > self.y_max:
                return None
            pp_track_x[y, x] = placement_type_mapped
        return pp_track_x, pp_track_id

    def preprocess_data(self, data_path):
        """
        Preprocesses data from JSON files located in the specified data_path directory.

        :param data_path: str
            The path to the directory containing JSON files.

        :return: list
            A list containing preprocessed data from the JSON files.

        :raises FileNotFoundError: If the specified data_path directory does not exist.
        :raises PermissionError: If the program does not have permission to access the data_path directory.

        Note:
        -----
        This method iterates through all JSON files in the specified directory,
        preprocesses each track within these files using the `preprocess_track` method,
        and collects the processed tracks into a list. Tracks that fail preprocessing
        are excluded from the final list.
        """
        self.data_path = data_path
        self.track_counter = 0
        self.file_counter = 0
        self.rejected_tracks_counter = 0
        x_list = []
        y_list = []
        for file in tqdm(os.listdir(self.data_path), f"Preprocessing data in {self.data_path}"):
            self.file_counter += 1
            if not file.endswith(".json"):
                continue
            with open(os.path.join(self.data_path, file), 'r') as f:
                json_data = json.load(f)
            for track in json_data:
                self.track_counter += 1
                processed_track = self.preprocess_track(track)
                if processed_track is not None:
                    pp_track_x, pp_track_id = processed_track
                    x_list.append(pp_track_x)
                    y_list.append(pp_track_id)
                    self.n_samples += 1
                    if self.n_samples >= self.n_max_samples:
                        return x_list, y_list
                else:
                    self.rejected_tracks_counter += 1
        return x_list, y_list

    def info(self):

        """
        Print detailed information about the current state of the Preprocessor object.

        Returns: None

        This method prints detailed information about the current state of the Preprocessor object,
        including the values of all its attributes.
        """
        return pprint(vars(self))
