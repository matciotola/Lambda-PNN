import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as io

from tools import utils
from tools.input_prepocessing import input_preparation


class MatDataset (Dataset):

    def __init__(self, img_path_list, sensor, device, ratio=4, semi_width=8, nbits=11):
        self.img_path_list = img_path_list
        self.ratio = ratio
        self.sensor = sensor
        self.nbits = nbits
        self.device = device
        self.semi_width = semi_width

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        temp = io.loadmat(self.img_path_list[index])
        pan = temp['I_PAN'].astype(np.float32)
        ms = temp['I_MS_LR'].astype(np.float32)
        with torch.no_grad():
            inputs, _, _, _ = input_preparation(ms, pan, self.ratio, self.nbits, self.device)
            threshold = utils.local_corr_mask(inputs,
                                              self.ratio,
                                              self.sensor,
                                              self.device,
                                              self.semi_width)

        return torch.squeeze(inputs, 0), torch.squeeze(threshold, 0)
