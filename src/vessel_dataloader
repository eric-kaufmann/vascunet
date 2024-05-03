import torch
from torch.utils.data import Dataset, DataLoader
from utils.vessel_iterator import VesselIterator
#from ..docs.config import DATA_DIR
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database")

class VesselDataset(Dataset):
    def __init__(self, data_path):
        self.vessel_iterator = VesselIterator(data_path)

    def __len__(self):
        return len(self.vessel_iterator)

    def __getitem__(self, idx):
        sample = self.vessel_iterator[idx]
        # Process the sample if needed
        # ...

        return sample

# Create an instance of the VesselDataset
dataset = VesselDataset(DATA_DIR)

# Set the batch size and other dataloader parameters
batch_size = 32
shuffle = True
num_workers = 4

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)