from multiprocessing import freeze_support
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vessel_iterator import VesselIterator
#from ..docs.config import DATA_DIR
from pathlib import Path
import utils.helper_functions as hf
import numpy as np


class VesselDataset(Dataset):
    def __init__(self, data_path):
        self.vessel_iterator = VesselIterator(data_path)
        self.v_index = 0

    def __len__(self):
        return len(self.vessel_iterator[self.v_index]['data_points'])

    def __getitem__(self, idx, v_idx=0, n_rand=10000):
        v = self.vessel_iterator[self.v_index]
        
        mesh_points = hf.move_and_rescale_matrix(v['mesh_points'])
        data_points = hf.move_and_rescale_matrix(v['data_points'])
        rand_idx = hf.pick_n_random_indices(mesh_points, n_rand)
        
        data = torch.tensor(
            np.concatenate(([data_points[idx]], mesh_points[rand_idx])),
            dtype=torch.float32
        )
        label = torch.tensor(
            v['velocity_systolic'][idx], 
            dtype=torch.float32
        )

        return data.T, label

if __name__ == "__main__":
    DATA_DIR = Path(r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database")
    freeze_support()
    dataset = VesselDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, batch in enumerate(dataloader):
        #print(f"Batch {i}: {batch.shape}")
        print(batch[0].shape, batch[1].shape)
        if i == 0:
            break