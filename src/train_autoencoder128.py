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
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--job_id', type=int, default=0, help='SLURM job ID')
    parser.add_argument('--num_epochs', type=int, default=100, help='Max number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    VECTOR_INPUT = args.vector_input
    LEARNING_RATE = args.learning_rate
else:
    VECTOR_INPUT = True
    LEARNING_RATE = 1e-6

class VesselGrid(Dataset):
    def __init__(self, folder_path):
        super(VesselGrid, self).__init__()
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        vessel_mask = torch.from_numpy(data['vessel_mask']).float()
        interpolated_velocities = torch.from_numpy(data['interpolated_velocities']).float().permute(3, 0, 1, 2)
        return vessel_mask.unsqueeze(0), interpolated_velocities


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
        
        # encoder
        #print("input shape",x.shape)
        x = encoder(x)
        #print("after encoder shape",x.shape)
        
        # add conditioning variables to feature vector
        conditioning_variables = torch.Tensor([])
        conditioning_variables = conditioning_variables.to(input_tensor.device)

        x = torch.concat([self.flatten(x), conditioning_variables], axis=1)
        
        # put new feature vector through the after_cond_encoder
        #print("x shape",x.shape)
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
        vessel_mask, interpolated_velocities = batch

        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities)
        else:
            x_hat, mu, log_var = self(vessel_mask)

        x_hat = x_hat * vessel_mask
        recon_loss = self.loss_fn(x_hat, interpolated_velocities)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        vessel_mask, interpolated_velocities = batch
        
        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities)
        else:
            x_hat, mu, log_var = self(vessel_mask)

        x_hat = x_hat * vessel_mask
        recon_loss = self.loss_fn(x_hat, interpolated_velocities)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        vessel_mask, interpolated_velocities = batch
        
        if VECTOR_INPUT:
            x_hat, mu, log_var = self(interpolated_velocities)
        else:
            x_hat, mu, log_var = self(vessel_mask)

        x_hat = x_hat * vessel_mask
        
        mse_accuracy = hf.calculate_mse_accuracy(x_hat, interpolated_velocities)
        angle_accuracy = hf.calculate_angle_between_tensors(x_hat, interpolated_velocities)
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

conditioning = nn.Sequential(
    nn.Linear(131072, 20),
    nn.ReLU()
)

conditioning2 = nn.Sequential(
    nn.Linear(10, 131072),
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
    
    torch.set_float32_matmul_precision('high')

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        after_cond_encoder=conditioning,
        pre_cond_decoder=conditioning2,
    )
    
    
    DATA_ROOT_DIR = hf.get_project_root() / "data" 
    DATA_DIR = "grid_vessel_data128_center"
    
    dataset = VesselGrid(DATA_ROOT_DIR / DATA_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    logger = TensorBoardLogger("./lightning_logs", name="autoencoder128")
    log_every = 50
    
    print(f"Model: {vae}")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs, 
        logger=logger, 
        log_every_n_steps=log_every, 
        callbacks=[TQDMProgressBar(refresh_rate=log_every)]
    )

    trainer.fit(vae, train_dataloader, val_dataloader)
    
    trainer.test(vae, val_dataloader)
    
    logging_path = hf.get_project_root() / "lightning_logs" / "autoencoder128" / f"version_{trainer.logger.version}"
    
    mse_acc = hf.load_tensorboard_data(str(logging_path), "mse_accuracy")
    norm_acc = hf.load_tensorboard_data(str(logging_path), "norm_accuracy")
    angle_acc = hf.load_tensorboard_data(str(logging_path), "angle_accuracy")
    
    # print(
    #     args.job_id,
    #     trainer.logger.version,
    #     DATA_DIR,
    #     VECTOR_INPUT,
    #     trainer.max_epochs,
    #     args.batch_size,
    #     LEARNING_RATE,
    #     mse_acc, 
    #     norm_acc, 
    #     angle_acc, 
    #     sep='\n'
    # )
    
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