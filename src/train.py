import lightning as L
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar


import utils.helper_functions as hf
from vessel_dataloader import VesselDataset
from models import NSModel, VesselGeomEmbedding
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--num_random_mesh_iterations', type=int, default=1, help='Number of random mesh iterations')
    parser.add_argument('--num_fluid_samples', type=int, default=20000, help='Number of fluid samples')
    parser.add_argument('--num_meshpoints', type=int, default=8192, help='Number of mesh points')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Train validation split')
    parser.add_argument('--num_neighbours', type=int, default=32, help='Number of neighbors')
    parser.add_argument('--num_midpoints', type=int, default=32, help='Number of midpoints')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--save_model', type=bool, default=True, help='Whether to save the model or not')
    args = parser.parse_args()
    return args

args = parse_arguments()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
TRAIN_VAL_SPLIT = args.train_val_split
NUM_NEIGHBOURS = args.num_neighbours
NUM_MIDPOINTS = args.num_midpoints
NUM_EPOCHS = args.num_epochs
DATA_DIR = hf.get_project_root() / "data" / "carotid_flow_database"
MODEL_DIR = hf.get_project_root() / "saved_models"
NUM_RANDOM_MESH_ITERATIONS = args.num_random_mesh_iterations
NUM_FLUID_SAMPLES = args.num_fluid_samples
NUM_MESHPOINTS = args.num_meshpoints
SEED = args.seed
SAVE = args.save_model



class VesselModel(L.LightningModule):
    def __init__(self, ns_model, vessel_geom_embedding_model, num_neighbours=32, num_midpoints=32, learning_rate=1e-3):
        super().__init__()
        self.ns_model = ns_model
        self.vessel_geom_embedding_model = vessel_geom_embedding_model
        self.num_neighbours = num_neighbours
        self.num_midpoints = num_midpoints
        self.learning_rate = learning_rate
        #self.scheduler = None

    def training_step(self, batch, batch_idx):
        mesh = batch['mesh_tensor']
        fluid_points = batch['fluid_points']
        sys_vel = batch['sys_vel_tensor']
        
        # get num_midpoints points index from farthest point sample which create center of patches
        fps_index = hf.farthest_point_sample(mesh, self.num_midpoints)
        # index to real points 
        mid_points = hf.index_points(mesh, fps_index)
        # calculate distances between the mesh_points and the new_points
        dists = hf.square_distance(mid_points, mesh)
        # get the indices of the num_neighbours closest points
        dist_idx = dists.argsort()[:, :, :self.num_neighbours]
        # group the points by the indices from dist_idx
        grouped_points = hf.index_points(mesh, dist_idx)
        # flatten the grouped points
        grouped_points_flat = torch.flatten(grouped_points, start_dim=1)
        
        mesh_emb = self.vessel_geom_embedding_model(grouped_points_flat)
        
        mesh_sample_input = torch.concat((fluid_points, mesh_emb), dim=1)
        
        ns_pred = self.ns_model(mesh_sample_input)

        loss = nn.functional.mse_loss(ns_pred, sys_vel)

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        #self.log("learning_rate", self.scheduler.get_last_lr(), prog_bar=False, on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mesh = batch['mesh_tensor']
        fluid_points = batch['fluid_points']
        sys_vel = batch['sys_vel_tensor']
        
        # get num_midpoints points index from farthest point sample which create center of patches
        fps_index = hf.farthest_point_sample(mesh, self.num_midpoints)
        # index to real points 
        mid_points = hf.index_points(mesh, fps_index)
        # calculate distances between the mesh_points and the new_points
        dists = hf.square_distance(mid_points, mesh)
        # get the indices of the num_neighbours closest points
        dist_idx = dists.argsort()[:, :, :self.num_neighbours]
        # group the points by the indices from dist_idx
        grouped_points = hf.index_points(mesh, dist_idx)
        # flatten the grouped points
        grouped_points_flat = torch.flatten(grouped_points, start_dim=1)
        
        mesh_emb = self.vessel_geom_embedding_model(grouped_points_flat)
        
        mesh_sample_input = torch.concat((fluid_points, mesh_emb), dim=1)
        
        ns_pred = self.ns_model(mesh_sample_input)

        loss = nn.functional.mse_loss(ns_pred, sys_vel)

        self.log("val_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # self.scheduler = scheduler
        return optimizer#, scheduler

def main():
    torch.set_float32_matmul_precision('medium')

    dataset = VesselDataset(
        DATA_DIR, 
        num_random_mesh_iterations=NUM_RANDOM_MESH_ITERATIONS, 
        num_fluid_samples=NUM_FLUID_SAMPLES, 
        num_meshpoints=NUM_MESHPOINTS, 
        seed=SEED,
        shuffle=True,
        validation=False,
        train_val_split=TRAIN_VAL_SPLIT
    )
    
    val_dataset = VesselDataset(
        DATA_DIR, 
        num_random_mesh_iterations=NUM_RANDOM_MESH_ITERATIONS, 
        num_fluid_samples=NUM_FLUID_SAMPLES, 
        num_meshpoints=NUM_MESHPOINTS, 
        seed=SEED,
        shuffle=True,
        validation=True,
        train_val_split=TRAIN_VAL_SPLIT
    )
    
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        persistent_workers=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        persistent_workers=True,
    )
    
    
    vessel_geom_embedding_model = VesselGeomEmbedding(
        in_features=NUM_MIDPOINTS*NUM_NEIGHBOURS*3, 
        out_features=256
    )
    
    ns_model = NSModel(
        in_features=256+3, 
        out_features=3
    )
    
    model = VesselModel(ns_model, vessel_geom_embedding_model, num_neighbours=NUM_NEIGHBOURS, num_midpoints=NUM_MIDPOINTS, learning_rate=LEARNING_RATE)
    
    logger = TensorBoardLogger("./lightning_logs", name="train")

    print("##################################") 
    print("len(dataset)",len(dataset))
    print("len(val_dataset)",len(val_dataset))
    print("train_val_split", TRAIN_VAL_SPLIT)
    print("data_dir", DATA_DIR)
    print("batch_size", BATCH_SIZE)
    print("learning_rate", LEARNING_RATE)
    print("random_mesh_iterations", NUM_RANDOM_MESH_ITERATIONS)
    print("num_fluid_samples", NUM_FLUID_SAMPLES)
    print("num_meshpoints", NUM_MESHPOINTS)
    print("seed", SEED)
    print("num_midpoints", NUM_MIDPOINTS)
    print("num_neighbours", NUM_NEIGHBOURS)
    print("tensorboard_logger dir", logger.log_dir)
    print("##################################") 
    
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS, 
        logger=logger, 
        log_every_n_steps=500,
        callbacks=[TQDMProgressBar(refresh_rate=2000)]
    )
    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    
    if SAVE:
        version_name = f"version_{trainer.logger.version}"
        torch.save(model.state_dict(), MODEL_DIR / version_name)


if __name__ == "__main__":
    main()