import lightning as L
import torch
from torch import optim, nn

class VesselModel(L.LightningModule):
    def __init__(self, model=None, input_size=10001*3):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 10000),
                nn.ReLU(),
                nn.Linear(10000, 10000),
                nn.ReLU(),
                nn.Linear(10000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.Linear(10, 3),
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        #x = x.T
        #y = y.view(-1, 1)

        x_hat = self.model(x)
        loss = nn.functional.mse_loss(x_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
