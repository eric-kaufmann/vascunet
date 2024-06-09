import lightning as L
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar


import utils.helper_functions as hf
from vessel_dataloader import VesselDataset1, VesselDatasetSinglePoint
from models import VesselModel1, VesselModel2, VesselModelSinlgePoint1
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--num_points', type=int, default=8192, help='Number of mesh points')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--save_model', type=bool, default=True, help='Whether to save the model or not')
    parser.add_argument('--model', type=str, default='VesselModel2', help='Model to use')
    parser.add_argument('--dataset', type=str, default='VesselDataset2', help='Dataset to use')
    args = parser.parse_args()
    return args

args = parse_arguments()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
DATA_DIR = hf.get_project_root() / "data" / "carotid_flow_database"
MODEL_DIR = hf.get_project_root() / "saved_models"
NUM_POINTS = args.num_points
SEED = args.seed
SAVE = args.save_model
VESSEL_MODEL = args.model
VESSEL_DATASET = args.dataset

def main():
    torch.set_float32_matmul_precision('medium')
    
    if VESSEL_DATASET == 'VesselDataset1':
        dataset = VesselDataset1(
            DATA_DIR, 
            num_points=NUM_POINTS,
            seed=SEED,
        )
    elif VESSEL_DATASET == 'VesselDatasetSinglePoint':
        dataset = VesselDatasetSinglePoint(
            DATA_DIR, 
            num_fluid_samples=8_192,
            num_meshpoints=4_096,
            seed=SEED,
            save_tensors=False,
            load_tensors=True,
        )
    else:
        raise ValueError("Dataset not found")
    
    # use 20% of training data for validation
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(SEED)
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    
    train_dataloader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
    )
    
    valid_dataloader = DataLoader(
        valid_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
    )
    
    
    ## Define model
    if VESSEL_MODEL == 'VesselModel1':
        model = VesselModel1(learning_rate=LEARNING_RATE, in_features=NUM_POINTS, out_features=NUM_POINTS, batch_size=BATCH_SIZE)
    elif VESSEL_MODEL == 'VesselModel2':
        model = VesselModel2(learning_rate=LEARNING_RATE, in_features=NUM_POINTS, out_features=NUM_POINTS, batch_size=BATCH_SIZE)
    elif VESSEL_MODEL == 'VesselModelSinglePoint1':
        model = VesselModelSinlgePoint1(learning_rate=LEARNING_RATE, in_features=4_096+1, out_features=3, batch_size=BATCH_SIZE)
    else:
        raise ValueError("Model not found")
    
    
    ## Define logger
    logger = TensorBoardLogger("./lightning_logs", name="train2")

    print("##################################") 
    print("len(dataset)",len(dataset))
    print("data_dir", DATA_DIR)
    print("batch_size", BATCH_SIZE)
    print("learning_rate", LEARNING_RATE)
    print("num_meshpoints", NUM_POINTS)
    print("seed", SEED)
    print("tensorboard_logger dir", logger.log_dir)
    print(model)
    print("##################################") 
    
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS, 
        logger=logger, 
        log_every_n_steps=500,
        callbacks=[TQDMProgressBar(refresh_rate=500)]
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    
    if SAVE:
        version_name = f"version_{trainer.logger.version}"
        torch.save(model.state_dict(), MODEL_DIR / version_name + ".pt")


if __name__ == "__main__":
    main()