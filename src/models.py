import lightning as L
import torch
from torch import optim, nn

class VesselModel1(L.LightningModule):
    def __init__(self, learning_rate=1e-3, in_features=2**14, out_features=2**14, batch_size=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_features*3),
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.out_features = out_features

    def training_step(self, batch, batch_idx): 
        points, sys_vel = batch      
        pred = self.model(points)
        pred = pred.reshape(sys_vel.shape)
        #loss = nn.functional.mse_loss(pred, sys_vel)
        loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss    
        
    def validation_step(self, batch, batch_idx):
        points, sys_vel = batch
        pred = self.model(points)
        pred = pred.reshape(sys_vel.shape)
        #loss = nn.functional.mse_loss(pred, sys_vel)
        loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def forward(self, x):
        x = torch.stack([x])
        x = self.model(x)
        x = x.reshape((self.out_features, 3))
        return x.detach()


class VesselModel2(L.LightningModule):
    def __init__(self, learning_rate=1e-3, in_features=2**14, out_features=2**14, batch_size=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features*3, 2**10),
            nn.Tanh(),
            nn.Linear(2**10, 2**10),
            nn.Tanh(),
            nn.Linear(2**10, 2**10),
            nn.Tanh(),
            nn.Linear(2**10, 2**10),
            nn.Tanh(),
            nn.Linear(2**10, out_features*3),
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.out_features = out_features

    def training_step(self, batch, batch_idx): 
        points, sys_vel = batch      
        pred = self.model(points)
        pred = pred.reshape(sys_vel.shape)
        loss = nn.functional.mse_loss(pred, sys_vel)
        #loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss    
        
    def validation_step(self, batch, batch_idx):
        points, sys_vel = batch
        pred = self.model(points)
        pred = pred.reshape(sys_vel.shape)
        loss = nn.functional.mse_loss(pred, sys_vel)
        #loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def forward(self, x):
        x = torch.stack([x])
        x = self.model(x)
        x = x.reshape((self.out_features, 3))
        return x.detach()
    
    
class VesselModelSinlgePoint1(L.LightningModule):
    def __init__(self, learning_rate=1025, in_features=3, out_features=2**14, batch_size=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, out_features),
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.out_features = out_features

    def training_step(self, batch, batch_idx): 
        mesh_points = batch['mesh_tensor']
        fluid_points = batch['fluid_points']
        sys_vel = batch['sys_vel_tensor']

        input_tensor = torch.concatenate((fluid_points.reshape(-1, 1, 3), mesh_points), dim=1) 
        pred = self.model(input_tensor)
        loss = nn.functional.mse_loss(pred, sys_vel)
        #loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss    
        
    def validation_step(self, batch, batch_idx):
        mesh_points = batch['mesh_tensor']
        fluid_points = batch['fluid_points']
        sys_vel = batch['sys_vel_tensor']

        input_tensor = torch.concatenate((fluid_points.reshape(-1, 1, 3), mesh_points), dim=1) 
        pred = self.model(input_tensor)
        loss = nn.functional.mse_loss(pred, sys_vel)
        #loss = torch.sum(torch.abs(pred - sys_vel))
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def forward(self, x):
        x = torch.stack([x])
        x = self.model(x)
        x = x.reshape((self.out_features, 3))
        return x.detach()