import torch
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, data_v, time_slot, predict_slot, batch_size):
        # super(dataset, self).__init__()
        self.time_slot = time_slot
        self.predict_slot = predict_slot
        self.data_v = data_v
        self.length = int((self.data_v.shape[0] - time_slot - predict_slot - 1) // batch_size * batch_size)

    def __getitem__(self, index):
        node_features = torch.tensor(self.data_v[index: index + self.time_slot], dtype=torch.float)
        _data = torch.reshape(node_features, (node_features.shape[-1],node_features.shape[0]))
        index += self.time_slot
        label_ = torch.tensor(self.data_v[index: index + self.predict_slot], dtype=torch.float)
        _label = torch.reshape(label_, (label_.shape[-1], label_.shape[0]))
        return _data, _label

    def __len__(self):
        return self.length