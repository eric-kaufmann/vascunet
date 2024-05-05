import lightning as L
from pathlib import Path

from vessel_dataloader import VesselDataset
from model import VesselModel

def main():
    DATA_DIR = Path(r"C:\Users\Eric Kaufmann\workspace\MA\data\carotid_flow_database")

    dataset = VesselDataset(DATA_DIR)
    model = VesselModel()

    trainer = L.Trainer(limit_train_batches=1000, max_epochs=5)
    trainer.fit(model=model, train_dataloaders=dataset)


if __name__ == "__main__":
    main()