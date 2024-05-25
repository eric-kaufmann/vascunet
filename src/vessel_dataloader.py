import torch
from torch.utils.data import Dataset
import os
import meshio
import numpy as np

class VesselDataset(Dataset):
    """
    A PyTorch dataset for loading vessel data.

    Parameters:
    - data_path (str): The path to the directory containing the vessel data.
    - samples_per_vessel (int): The number of samples to generate per vessel.
    - num_meshpoints (int): The number of mesh points to sample per sample.
    - seed (int): The random seed for reproducibility.

    Attributes:
    - data_path (str): The path to the directory containing the vessel data.
    - samples_per_vessel (int): The number of samples to generate per vessel.
    - vessel_files (list): A list of dictionaries, where each dictionary contains the paths to the fluid and mesh files.
    - num_meshpoints (int): The number of mesh points to sample per sample.
    - current_vessel (int): The index of the current vessel being processed.
    - current_vessel_tensor (torch.Tensor): The tensor representing the current vessel's mesh data.

    Methods:
    - __len__(): Returns the total number of samples in the dataset.
    - __getitem__(idx): Returns the sample at the given index.
    - _update_vessel_tensor(vessel_idx): Updates the current vessel tensor based on the given vessel index.
    - _get_vessel_files(): Retrieves the vessel files from the given directory.
    """

    def __init__(self, data_path, num_random_mesh_iterations=10_000, num_fluid_samples=50_000, num_meshpoints=8192, seed=None, shuffle=False, validation=False, train_val_split=0.8):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            torch.seed()
            np.random.seed()
            
        self.data_path = data_path
        self.samples_per_vessel = num_random_mesh_iterations
        vessel_files = self._get_vessel_files(shuffle)
        if validation:
            self.vessel_files = vessel_files[int(len(vessel_files)*train_val_split):]
        else: 
            self.vessel_files = vessel_files[:int(len(vessel_files)*train_val_split)]
        self.num_meshpoints = num_meshpoints
        self.num_fluid_samples = num_fluid_samples
        #self.shuffle = shuffle
    
        self.current_vessel = 0
        self.current_vessel_mesh_tensor = self._update_vessel_mesh_tensor(0)
        self.current_vessel_fluid_point_tensor = self._update_vessel_fluid_point_tensor(0)
        self.current_vessel_sys_vel_tensor = self._update_vessel_sys_vel_tensor(0)
    

    def __len__(self):
        return len(self.vessel_files) * self.samples_per_vessel * self.num_fluid_samples

    def __getitem__(self, idx):
        fluid_idx = idx % self.num_fluid_samples
        temp_idx = idx // self.num_fluid_samples
        vessel_idx = temp_idx // self.samples_per_vessel
        mesh_iter_idx = temp_idx % self.samples_per_vessel
        if vessel_idx != self.current_vessel:
            self.current_vessel = vessel_idx
            self.current_vessel_mesh_tensor = self._update_vessel_mesh_tensor(vessel_idx)
            self.current_vessel_sys_vel_tensor = self._update_vessel_sys_vel_tensor(vessel_idx)
            self.current_vessel_fluid_point_tensor = self._update_vessel_fluid_point_tensor(vessel_idx)
         
        return_dict = {
            'mesh_tensor': self.current_vessel_mesh_tensor[mesh_iter_idx],
            'fluid_points': self.current_vessel_fluid_point_tensor[fluid_idx],
            'sys_vel_tensor': self.current_vessel_sys_vel_tensor[fluid_idx],
        }    
        
        return return_dict
    
    def _update_vessel_mesh_tensor(self, vessel_idx):
        """
        Updates the current vessel tensor based on the given vessel index.

        Parameters:
        - vessel_idx (int): The index of the vessel.

        Returns:
        - torch.Tensor: The tensor representing the vessel's mesh data.
        """
        vessel = self.vessel_files[vessel_idx]
        mesh_data = meshio.read(vessel['mesh'])
        mesh_tensor = torch.Tensor(mesh_data.points)
        mesh_tensor = (mesh_tensor - mesh_tensor.min()) / (mesh_tensor.max() - mesh_tensor.min())
        random_sample_idx = torch.randint(0, mesh_tensor.shape[0], (self.samples_per_vessel,self.num_meshpoints))
        return mesh_tensor[random_sample_idx]
    
    def _update_vessel_fluid_point_tensor(self, vessel_idx):
        """
        Updates the current vessel tensor based on the given vessel index.

        Parameters:
        - vessel_idx (int): The index of the vessel.

        Returns:
        - torch.Tensor: The tensor representing the vessel's mesh data.
        """
        vessel = self.vessel_files[vessel_idx]
        fluid_data = meshio.read(vessel['fluid'])
        fluid_points = torch.Tensor(fluid_data.points)
        fluid_points = (fluid_points - fluid_points.min()) / (fluid_points.max() - fluid_points.min())
        return fluid_points
    
    def _update_vessel_sys_vel_tensor(self, vessel_idx):
        """
        Updates the current vessel tensor based on the given vessel index.

        Parameters:
        - vessel_idx (int): The index of the vessel.

        Returns:
        - torch.Tensor: The tensor representing the vessel's mesh data.
        """
        vessel = self.vessel_files[vessel_idx]
        fluid_data = meshio.read(vessel['fluid'])
        sys_vel = torch.Tensor(fluid_data.point_data['velocity_systolic'])
        sys_vel = (sys_vel - sys_vel.min()) / (sys_vel.max() - sys_vel.min())
        return sys_vel
    
    def _get_vessel_files(self, shuffle):
        """
        Retrieves the vessel files from the given directory.

        Returns:
        - list: A list of dictionaries, where each dictionary contains the paths to the fluid and mesh files.
        """
        vessel_files = []
        for fluid_filename in os.listdir(self.data_path):
            if fluid_filename.endswith('fluid.vtu'):
                fluid_file_path = os.path.join(self.data_path, fluid_filename)
                mesh_file_path = fluid_file_path.replace('fluid.vtu', 'wss.vtu')
                vessel_files.append({
                    'fluid': fluid_file_path,
                    'mesh': mesh_file_path
                })
        vessel_files = np.array(vessel_files)
        
        if shuffle:
            np.random.shuffle(vessel_files)
            
        return vessel_files
        