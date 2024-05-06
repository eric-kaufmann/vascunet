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
                nn.Linear(10000, 5000),
                nn.ReLU(),
                nn.Linear(5000, 3000),
                nn.ReLU(),
                nn.Linear(3000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
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
