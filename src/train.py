import lightning as L
from pathlib import Path

from vessel_dataloader import VesselDataset
from model import VesselModel
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database")
BATCH_SIZE = 8

def main():

    dataset = VesselDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = VesselModel()

    trainer = L.Trainer(max_epochs=5, limit_train_batches=200)
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()