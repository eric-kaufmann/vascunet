#!/home/ne34gux/workspace/vascunet/.venv/bin/python

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import pprint

import torch.nn as nn
import torch.optim as optim
import utils.helper_functions as hf
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--vector_input', action='store_true', help='Use vector input')
    parser.add_argument('--add_metadata', action='store_true', help='Use vector input')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--job_id', type=int, default=0, help='SLURM job ID')
    parser.add_argument('--num_epochs', type=int, default=100, help='Max number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--log_every', type=int, default=100, help='Log every n steps')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    VECTOR_INPUT = args.vector_input
    LEARNING_RATE = args.learning_rate
    ADD_METADATA = args.add_metadata
    METADATA_SIZE = 18
    
else:
    VECTOR_INPUT = True
    LEARNING_RATE = 1e-6
    ADD_METADATA = True
    METADATA_SIZE = 18
    
METADATA_PATH = "/home/ne34gux/workspace/vascunet/data/grid_vessel_metadata"

class VesselGrid(Dataset):
    def __init__(self, folder_path, add_metadata=False):
        super(VesselGrid, self).__init__()
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.add_metadata = add_metadata
        if add_metadata:
            self.metadata_file_paths = [os.path.join(METADATA_PATH, f) for f in os.listdir(METADATA_PATH) if f.endswith('.npz')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data_file_path = self.file_paths[idx]
        data = np.load(data_file_path)
        vessel_mask = torch.from_numpy(data['vessel_mask']).float()
        interpolated_velocities = torch.from_numpy(data['interpolated_velocities']).float().permute(3, 0, 1, 2)
        
        if self.add_metadata:
            vessel_name = data_file_path.split('_angle_')[0][-10:]
            metadata = np.load(os.path.join(METADATA_PATH, vessel_name+'.npz'))
            metadata_tensor = torch.concatenate([
                torch.from_numpy(metadata['dimension_min']),
                torch.from_numpy(metadata['dimension_max']),
                torch.from_numpy(metadata['eigenvectors']).flatten(),
                torch.from_numpy(metadata['eigenvalues']),
            ])
            return vessel_mask.unsqueeze(0), interpolated_velocities, metadata_tensor.float()
        else:
            return vessel_mask.unsqueeze(0), interpolated_velocities


class VAE(pl.LightningModule):
    def __init__(self, encoder, decoder, after_cond_encoder, pre_cond_decoder, batch_size=1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.after_cond_encoder = after_cond_encoder
        self.pre_cond_decoder = pre_cond_decoder
        self.batch_size = batch_size
        
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.flatten = nn.Flatten()
    
    def forward(self, input_tensor, metadata=None):   
        x = input_tensor
        
        x = encoder(x)
        
        if ADD_METADATA:
            conditioning_variables = torch.Tensor(metadata)
        else:
            conditioning_variables = torch.Tensor([])

        x = torch.concatenate([self.flatten(x), conditioning_variables], axis=1)
        
        x = self.after_cond_encoder(x)
        
        # reparameterize
        mu, log_var = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # add conditioning variables to z
        z = torch.concat([z, conditioning_variables], axis=1)
        
        # put z through the pre_cond_decoder
        z = self.pre_cond_decoder(z)
        
        # reshape z to block shape
        z = torch.reshape(z, (self.batch_size, 256, 8, 8, 8))
        
        # put new block shaped z through the decoder
        #print("before",z.shape)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[0:3](z)
        z = decoder[3:6](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[6:9](z)
        z = decoder[9:12](z)
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)
        z = decoder[12:](z)
        #print("after", z.shape)
        return z, mu, log_var
    
    def training_step(self, batch, batch_idx):
        if ADD_METADATA:
            vessel_mask, interpolated_velocities, metadata = batch
        else:
            vessel_mask, interpolated_velocities = batch
            metadata = None

        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities, metadata=metadata)
        else:
            x_hat, mu, log_var = self(vessel_mask, metadata=metadata)
            
        x_hat = x_hat * vessel_mask
        interpolated_velocities = interpolated_velocities * vessel_mask
        # if torch.isnan(x_hat).any():
        #     print("x_hat contains NaNs")
        
        angle_loss = torch.mean(hf.calculate_angles_from_grid(x_hat, interpolated_velocities))
        recon_loss = self.loss_fn(x_hat, interpolated_velocities)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # print(f"angle_loss: {angle_loss}")
        # print(f"recon_loss: {recon_loss}")
        # print(f"kl_loss: {kl_loss}")
        # # Debugging: Check for NaNs in angle_loss
        # if torch.isnan(angle_loss).any():
        #     print("NaN detected in angle_loss")
        #     print(f"x_hat has NAN?: {torch.isnan(x_hat).any()}")
        #     print(f"interpolated_velocities has NAN?: {torch.isnan(interpolated_velocities).any()}")
        #     print(f"angle_loss has NAN?: {torch.isnan(angle_loss).any()}")
        #     exit()
            
        #print(angle_loss, x_hat.shape, interpolated_velocities.shape, hf.calculate_angle_between_tensors(x_hat, interpolated_velocities).shape)
        
        loss = recon_loss + kl_loss + angle_loss.item()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_angle_loss', angle_loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if ADD_METADATA:
            vessel_mask, interpolated_velocities, metadata = batch
        else:
            vessel_mask, interpolated_velocities = batch
            metadata = None

        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities, metadata=metadata)
        else:
            x_hat, mu, log_var = self(vessel_mask, metadata=metadata)
            
        x_hat = x_hat * vessel_mask
        
        angle_loss = torch.mean(hf.calculate_angles_from_grid(x_hat, interpolated_velocities))
        recon_loss = self.loss_fn(x_hat, interpolated_velocities)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = recon_loss + kl_loss + angle_loss.item()
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        self.log('val_angle_loss', angle_loss.item())

        return loss
    
    def test_step(self, batch, batch_idx):
        if ADD_METADATA:
            vessel_mask, interpolated_velocities, metadata = batch
        else:
            vessel_mask, interpolated_velocities = batch
            metadata = None
        
        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities, metadata=metadata)
        else:
            x_hat, mu, log_var = self(vessel_mask, metadata=metadata)

        x_hat = x_hat * vessel_mask
        
        mse_accuracy = hf.calculate_mse_accuracy(x_hat, interpolated_velocities)
        angle_accuracy = hf.calculate_angles_from_grid(x_hat, interpolated_velocities, deg_output=True)
        norm_accuracy = hf.calculate_difference_norm(x_hat, interpolated_velocities)

        self.log('mse_accuracy', mse_accuracy)
        self.log('angle_accuracy', angle_accuracy.mean())
        self.log('norm_accuracy', norm_accuracy.mean())
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, encoder, decoder, after_cond_encoder, pre_cond_decoder, batch_size=1):
        model = VAE(encoder, decoder, after_cond_encoder, pre_cond_decoder, batch_size)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
   
encoder_input_size = 3 if VECTOR_INPUT else 1   
    
encoder = nn.Sequential(
    nn.Conv3d(encoder_input_size, 3, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
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

metadata_vector_size = METADATA_SIZE if ADD_METADATA else 0

conditioning = nn.Sequential(
    nn.Linear(131072+metadata_vector_size, 20),
    nn.ReLU()
)

conditioning2 = nn.Sequential(
    nn.Linear(10+metadata_vector_size, 131072),
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
    nn.BatchNorm3d(3),
    nn.ReLU(inplace=True),
    nn.ConvTranspose3d(3, 3, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm3d(3),
)

if __name__ == "__main__":
    
    torch.set_float32_matmul_precision('medium')

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        after_cond_encoder=conditioning,
        pre_cond_decoder=conditioning2,
    )
    
    
    DATA_ROOT_DIR = hf.get_project_root() / "data" 
    DATA_DIR = "grid_vessel_data128_center"
    
    dataset = VesselGrid(DATA_ROOT_DIR / DATA_DIR, add_metadata=ADD_METADATA)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    logger = TensorBoardLogger("./lightning_logs", name="autoencoder128")
    
    print(f"Model: {vae}")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs, 
        logger=logger, 
        log_every_n_steps=args.log_every, 
        callbacks=[TQDMProgressBar(refresh_rate=args.log_every)]
    )

    trainer.fit(vae, train_dataloader, val_dataloader)
    
    trainer.test(vae, val_dataloader)
    
    logging_path = hf.get_project_root() / "lightning_logs" / "autoencoder128" / f"version_{trainer.logger.version}"
    
    mse_acc = hf.load_tensorboard_data(str(logging_path), "mse_accuracy")
    norm_acc = hf.load_tensorboard_data(str(logging_path), "norm_accuracy")
    angle_acc = hf.load_tensorboard_data(str(logging_path), "angle_accuracy")
    
    print("##################################################")
    print("##################################################")
    print()
    print("#### METRICS ####")
    
    metrics = {
        'SlurmID': args.job_id,
        'LoggingVersion': trainer.logger.version,
        'DataPath': str(DATA_DIR),
        'isVectorInput': VECTOR_INPUT,
        'NumEpochs': args.num_epochs,
        'BatchSize': args.batch_size,
        'LearningRate': LEARNING_RATE,
        'MSEAccuracy': mse_acc[0],
        'AverageNormDifference': norm_acc[0],
        'AverageAngleDifference': angle_acc[0]
    }
    
    pprint.pprint(metrics)
    
    print("Write metrics into results.csv")
    
    hf.add_metrics_to_result_file(metrics)