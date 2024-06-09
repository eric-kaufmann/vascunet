import torch
from torch.utils.data import Dataset
import os
import meshio
import numpy as np

class VesselDatasetSinglePoint(Dataset):
    def __init__(self, data_path, num_fluid_samples=1024, num_meshpoints=1024, seed=None, save_tensors=False, load_tensors=False):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            torch.seed()
            np.random.seed()
            
        self.data_path = data_path
        self.vessel_files = self._get_vessel_files()
        self.num_vessels = len(self.vessel_files)
        self.num_fluid_samples = num_fluid_samples
        
        if save_tensors:
            self.mesh_points_tensor, self.fluid_points_tensor, self.sys_vel_tensor = self._get_vessel_tensor(num_meshpoints, num_fluid_samples)
            torch.save(self.mesh_points_tensor, '/home/ne34gux/workspace/vascunet/data/torch_tensors/mesh_points.pt')
            torch.save(self.fluid_points_tensor, '/home/ne34gux/workspace/vascunet/data/torch_tensors/fluid_points.pt')
            torch.save(self.sys_vel_tensor, '/home/ne34gux/workspace/vascunet/data/torch_tensors/sys_vel.pt')
            
        if load_tensors:
            self.mesh_points_tensor = torch.load('/home/ne34gux/workspace/vascunet/data/torch_tensors/mesh_points.pt')
            self.fluid_points_tensor = torch.load('/home/ne34gux/workspace/vascunet/data/torch_tensors/fluid_points.pt')
            self.sys_vel_tensor = torch.load('/home/ne34gux/workspace/vascunet/data/torch_tensors/sys_vel.pt')
        else:
            self.mesh_points_tensor, self.fluid_points_tensor, self.sys_vel_tensor = self._get_vessel_tensor(num_meshpoints, num_fluid_samples)

    def __len__(self):
        return self.num_vessels * self.num_fluid_samples

    def __getitem__(self, idx):
        vessel_idx = idx % self.num_vessels
        sample_idx = idx // self.num_vessels

        return_dict = {
            'mesh_tensor': self._normalize_data(self.mesh_points_tensor[vessel_idx]),
            'fluid_points': self.fluid_points_tensor[vessel_idx,sample_idx],
            'sys_vel_tensor': self.sys_vel_tensor[vessel_idx,sample_idx]
        }    
        
        return return_dict
    
    def _get_vessel_tensor(self, num_mesh_points, num_fluid_points):
        mesh_points_list = []
        fluid_points_list = []
        sys_vel_list = []
        for vf in self.vessel_files:
            fluid_points = torch.Tensor(meshio.read(vf['fluid']).points)
            mesh_points = torch.Tensor(meshio.read(vf['mesh']).points)
            sys_vel = torch.Tensor(meshio.read(vf['fluid']).point_data['velocity_systolic'])
            
            idx1 = np.random.choice(np.arange(len(mesh_points)), num_mesh_points, replace=False)
            idx2 = np.random.choice(np.arange(len(fluid_points)), num_fluid_points, replace=False)
            
            mesh_points_list.append(mesh_points[idx1])
            fluid_points_list.append(fluid_points[idx2])
            sys_vel_list.append(sys_vel[idx2])

        mesh_points_tensor = torch.stack(mesh_points_list)
        fluid_points_tensor = torch.stack(fluid_points_list)
        sys_vel_tensor = torch.stack(sys_vel_list)
        
        return mesh_points_tensor, fluid_points_tensor, sys_vel_tensor
    
    def _get_vessel_files(self):
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
            
        return vessel_files
    
    def _normalize_data(self, tensor):
        tensor_min = tensor.min(dim=0, keepdim=True)[0]
        tensor_max = tensor.max(dim=0, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return tensor
        

class VesselDataset1(Dataset):

    def __init__(self, data_path, num_points=10_000, seed=None, apply_transformation=True):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            torch.seed()
            np.random.seed()
            
        self.data_path = data_path
        self.num_points = num_points
        self.vessel_files = self._get_vessel_files()
        self.apply_transformation = apply_transformation
    

    def __len__(self):
        return len(self.vessel_files)

    def __getitem__(self, idx): 
        raw_data = self._point_data_generator(self.vessel_files[idx])
        if self.apply_transformation:
            transformed_data = self._transform_data(raw_data)
        else:
            transformed_data = raw_data
        return transformed_data
    
    def _get_vessel_files(self):
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
        return vessel_files
        
    def _point_data_generator(self, vessel_file:dict):
        mesh_data = meshio.read(vessel_file['mesh'])
        fluid_data = meshio.read(vessel_file['fluid'])
        idx = np.random.choice(np.arange(fluid_data.points.shape[0]), self.num_points, replace=False)
        return (
            torch.concat([torch.Tensor(fluid_data.points), torch.Tensor(mesh_data.points)], axis=0)[idx],
            torch.concat([torch.Tensor(fluid_data.point_data['velocity_systolic']), torch.zeros(mesh_data.points.shape)], axis=0)[idx]
        )
        
    def _transform_data(self, raw_data):
        points, velocities = raw_data
        # Normalize points
        points_min = points.min(dim=0, keepdim=True)[0]
        points_max = points.max(dim=0, keepdim=True)[0]
        points = (points - points_min) / (points_max - points_min)
        
        transformed_data = (points, velocities)
        return transformed_data
        