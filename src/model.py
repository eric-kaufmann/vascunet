import lightning as L
from torch import optim, nn

class VesselModel(L.LightningModule):
    def __init__(self, model=None, input_size=10001):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
            )

    def training_step(self, batch):
        x, y = batch
        x = x.T
        y = y.view(-1, 1)

        x_hat = self.model(x)
        loss = nn.functional.mse_loss(x_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer