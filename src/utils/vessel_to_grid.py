import helper_functions as hf
import os
import numpy as np
import torch
import meshio

#(batch, size=(64, 64, 64), method='linear', threashold=0.1)

DATA_DIR = hf.get_project_root() / "data" / "carotid_flow_database"
SAVE_DIR = hf.get_project_root() / "data" / "grid_vessel_data"

print("Loading vessel files...")
vessel_files = []
for fluid_filename in os.listdir(DATA_DIR):
    if fluid_filename.endswith('fluid.vtu'):
        fluid_file_path = os.path.join(DATA_DIR, fluid_filename)
        mesh_file_path = fluid_file_path.replace('fluid.vtu', 'wss.vtu')
        vessel_files.append({
            'fluid': fluid_file_path,
            'mesh': mesh_file_path
        })
vessel_files = np.array(vessel_files)

print("Processing vessel files...")
for v_idx, v in enumerate(vessel_files):
    print(f"Processing vessel {v_idx + 1}/{len(vessel_files)}")
    
    mesh_data = meshio.read(v['mesh'])
    fluid_data = meshio.read(v['fluid'])
        
    data_points = torch.concat(
        [torch.Tensor(fluid_data.points), torch.Tensor(mesh_data.points)],
        axis=0
    )
    
    # trasform data
    points_min = data_points.min(dim=0, keepdim=True)[0]
    points_max = data_points.max(dim=0, keepdim=True)[0]
    data_points = (data_points - points_min) / (points_max - points_min)

    velocity_vectors = torch.concat(
        [torch.Tensor(fluid_data.point_data['velocity_systolic']), torch.zeros(mesh_data.points.shape)], 
        axis=0
    )
    
    vessel_mask, interpolated_velocities = hf.get_vessel_grid_data(
        (data_points, velocity_vectors), 
        size=(64, 64, 64), 
        method='linear', 
        threashold=0.1
    )
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    np.savez(os.path.join(SAVE_DIR, f'vessel_{v_idx}.npz'), vessel_mask=vessel_mask, interpolated_velocities=interpolated_velocities)
    
    
print("Processing complete.")
    
