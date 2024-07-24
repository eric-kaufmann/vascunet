import helper_functions as hf
import os
import numpy as np
import torch
import meshio
import time

#(batch, size=(64, 64, 64), method='linear', threashold=0.1)

DATA_DIR = hf.get_project_root() / "data" / "carotid_flow_database"
SAVE_DIR = hf.get_project_root() / "data" / "grid_vessel_data128_center"

THREASHOLD = 0
SIZE = 128
NUM_ANGLES = 8

start_time = time.time()

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
    #data_points = hf.normalize_point_cloud(data_points, fix_resizing_factor=False)
    data_points = hf.center_point_cloud_to_unit_cube(data_points)

    velocity_vectors = torch.concat(
        [torch.Tensor(fluid_data.point_data['velocity_systolic']), torch.zeros(mesh_data.points.shape)], 
        axis=0
    )
    
    _ , out_points = hf.check_points_in_hull(data_points, torch.rand(300_000, 3))
    
    data_points, velocity_vectors = torch.concat([data_points, out_points]), torch.concat([velocity_vectors, torch.zeros(out_points.shape)])
    
    for angle in np.linspace(0, 360, NUM_ANGLES):
        
        file_name = f"vessel_{v_idx:03d}_angle_{int(angle):03d}.npz"
        file_path = os.path.join(SAVE_DIR, file_name)
        
        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping...")
            continue
        
        #print(f"calc vessel_{v_idx:03d}_angle_{int(angle):03d}.npz")
        rotated_velocities = torch.from_numpy(hf.rotate_vector_field(velocity_vectors.numpy(), angle))
        rotated_mask = torch.from_numpy(hf.rotate_vector_field(data_points.numpy(), angle, center=[0.5,0.5,0]))
        
        
        vessel_mask, interpolated_velocities = hf.get_vessel_grid_data(
            (rotated_mask, rotated_velocities), 
            size=(SIZE, SIZE, SIZE), 
            method='linear', 
            threashold=THREASHOLD
        )
        
        print(f"\tsave {file_name}")
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        np.savez(
            file_path, 
            vessel_mask=vessel_mask, 
            interpolated_velocities=interpolated_velocities,
        )
        
    print(f"current duration: {time.time() - start_time} seconds")

end_time = time.time()
duration = end_time - start_time
hours, remainder = divmod(duration, 3600)
minutes, seconds = divmod(remainder, 60)
    
print("Processing complete.")
print(f"Loading and processing took {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds.")
    
