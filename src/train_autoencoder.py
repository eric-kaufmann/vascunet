import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy.interpolate import griddata

from vessel_dataloader import VesselAutoencoderDataset
import utils.helper_functions as hf

from torch.utils.data import DataLoader
import torch.nn.functional as F


def interpolate_vectors_to_grid(points, velocities, grid_shape, method='linear'):

    # Generate linearly spaced points for each axis based on the bounds and the desired shape
    x = np.linspace(np.min(points[:,0]), np.max(points[:,0]), grid_shape[0])
    y = np.linspace(np.min(points[:,1]), np.max(points[:,1]), grid_shape[1])
    z = np.linspace(np.min(points[:,2]), np.max(points[:,2]), grid_shape[2])

    # Create a 3D grid from the 1D arrays
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # Interpolate the velocity vectors onto the grid
    grid_vx = griddata(points, velocities[:,0], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    grid_vy = griddata(points, velocities[:,1], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    grid_vz = griddata(points, velocities[:,2], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    
    # Combine the interpolated velocities into a single array
    grid_velocities = np.stack((grid_vx, grid_vy, grid_vz), axis=-1)
    
    return grid_velocities
    
    
def get_vessel_grid_data(batch, size=(64, 64, 64), method='linear', threashold=0.1):
    points, velocities = batch
    interpolated_velocities = interpolate_vectors_to_grid(
        points.cpu().numpy(), 
        velocities.cpu().numpy(), 
        size, 
        method=method
    )
    vessel_mask = np.sum(interpolated_velocities**2, axis=-1) > threashold
    interpolated_velocities[vessel_mask == False] = 0
    return torch.Tensor(vessel_mask), torch.Tensor(interpolated_velocities)

class VAE(pl.LightningModule):
    def __init__(self, encoder, decoder, after_cond_encoder, pre_cond_decoder, batch_size=1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.after_cond_encoder = after_cond_encoder
        self.pre_cond_decoder = pre_cond_decoder
        self.batch_size = batch_size
        
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.flatten = nn.Flatten()
    
    def calculate_conditional_variables(self, input_tensor):
        return torch.Tensor([])
    
    def forward(self, input_tensor):
        
        x = input_tensor
        print("input", x.shape)
        
        # encoder
        x = encoder(x)
        print("after encoder", x.shape)
        
        # add conditioning variables to feature vector
        conditioning_variables = torch.Tensor([])
        x = torch.concat([self.flatten(input_tensor), conditioning_variables], axis=1)
        print("after conditioning", x.shape)
        
        # put new feature vector through the after_cond_encoder
        x = self.after_cond_encoder(x)
        print("after conditioning encoder", x.shape)
        
        # reparameterize
        mu, log_var = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        print("reparameterize", z.shape)

        # add conditioning variables to z
        z = torch.concat([z, conditioning_variables], axis=1)
        print("after conditioning2", z.shape)
        
        # put z through the pre_cond_decoder
        z = self.pre_cond_decoder(z)
        print("after conditioning decoder", z.shape)
        
        # reshape z to block shape
        z = torch.reshape(z, (self.batch_size, 256, 8, 8, 8))
        print("after reshaping", z.shape)
        
        # put new block shaped z through the decoder
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[0:3](z)
        z = decoder[3:6](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[6:9](z)
        z = decoder[9:12](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[12:](z)
        print("after decoder", z.shape)

        return z, mu, log_var
    
    def training_step(self, batch, batch_idx):
        #x, y = batch
        
        vessel_mask, interpolated_velocities = get_vessel_grid_data(batch, threashold=0.2, method='linear')
        #x = x.view(x.size(0), -1)
        x_hat, mu, log_var = self(vessel_mask)
        
        # Compute reconstruction loss
        recon_loss = self.loss_fn(x_hat, interpolated_velocities)
        
        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = recon_loss + kl_loss
        
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    
encoder = nn.Sequential(
    nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(16),
    nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(32),
    nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(64),
    nn.Conv3d(64, 96, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(96),
    nn.Conv3d(96, 128, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(128),
    nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(192),
    nn.Conv3d(192, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm3d(256),
)

conditioning_shape = (1, 256, 8, 8, 8)
conditioning_input_shape = int(torch.Tensor(conditioning_shape[1:]).prod())

conditioning = nn.Sequential(
    nn.Linear(262144, 10),
    nn.ReLU()
)

conditioning2 = nn.Sequential(
    nn.Linear(5, 131072),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.Conv3d(256, 192, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(192),
    nn.ReLU(inplace=True),
    nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(128),
    nn.ReLU(inplace=True),
    nn.Conv3d(128, 96, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(96),
    nn.ReLU(inplace=True),
    nn.Conv3d(96, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(64),
    nn.ReLU(inplace=True),
    nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(32),
    nn.ReLU(inplace=True),
    nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(16),
    nn.ReLU(inplace=True),
    nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm3d(3)
)

if __name__ == "__main__":
    # Create the VAE model
    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        after_cond_encoder=conditioning,
        pre_cond_decoder=conditioning2,
    )


    DATA_DIR = hf.get_project_root() / "data" / "carotid_flow_database"

    dataset = VesselAutoencoderDataset(
        DATA_DIR, 
        apply_point_restrictions=False
    )

    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=1, 
    #     shuffle=False, 
    #     num_workers=2, 
    # )
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Create a trainer
    trainer = pl.Trainer(max_epochs=1)

    # Train the VAE with the DataLoaders
    trainer.fit(vae, train_dataloader, val_dataloader)