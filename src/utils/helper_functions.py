import numpy as np
from pathlib import Path
import torch
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from tensorboard.backend.event_processing import event_accumulator
import os
import csv

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def pick_n_random_indices(arr, n):
    """
    Randomly selects n indices from the given array.

    Parameters:
    arr (array-like): The input array.
    n (int): The number of indices to select.

    Returns:
    numpy.ndarray: An array of n randomly selected indices from the input array.
    """
    return np.random.choice(np.arange(len(arr)-1), n, replace=False)

def move_and_rescale_matrix(matrix):
    """
    Move and rescale the given matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: The moved and rescaled matrix.
    """
    # Move the matrix to have minimum values at 0
    min_vals = np.min(matrix, axis=0)
    matrix -= min_vals

    # Rescale the matrix to have maximum values at 1
    max_vals = np.max(matrix, axis=0)
    matrix /= max_vals

    return matrix


def interpolate_vectors_to_grid(points, velocities, grid_shape, method='linear'):
    """
    Interpolates velocity vectors at 3D points into a 3D grid of a predefined shape.

    Parameters:
    - points: A numpy array of shape (N, 3), where N is the number of points, representing the 3D coordinates.
    - velocities: A numpy array of shape (N, 3), where N is the number of points, representing the velocity vectors at these points.
    - grid_shape: A tuple of 3 integers defining the shape of the 3D grid (depth, height, width).
    - method: Interpolation method. Options include 'linear', 'nearest', and 'cubic'.

    Returns:
    - A numpy array of shape (grid_shape[0], grid_shape[1], grid_shape[2], 3) representing the interpolated velocity vectors on the 3D grid.
    """
    
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
        np.array(points), 
        np.array(velocities), 
        size, 
        method=method
    )
    vessel_mask = np.sum(interpolated_velocities**2, axis=-1) > threashold
    interpolated_velocities[vessel_mask == False] = 0
    return torch.Tensor(vessel_mask), torch.Tensor(interpolated_velocities)

def normalize_point_cloud(data_points, fix_resizing_factor=True):
    points_min = data_points.min(dim=0, keepdim=True)[0]
    points_max = data_points.max(dim=0, keepdim=True)[0]

    if fix_resizing_factor:
        ranges = points_max - points_min
        max_range = ranges.max()
    else:
        max_range = (points_max - points_min)
    return (data_points - points_min) / max_range

def check_points_in_hull(hull_tensor, check_tensor):
    hull_points_np = hull_tensor.numpy()
    check_points_np = check_tensor.numpy()
    
    hull = ConvexHull(hull_points_np)
    
    delaunay = Delaunay(hull_points_np[hull.vertices])
    
    inside = delaunay.find_simplex(check_points_np) >= 0
    
    inside_points = check_points_np[inside]
    outside_points = check_points_np[~inside]
    
    inside_tensor = torch.from_numpy(inside_points)
    outside_tensor = torch.from_numpy(outside_points)
    
    return inside_tensor, outside_tensor

def reshape_tensor(tensor):
    if tensor.dim() != 4:
        raise ValueError("Input tensor must have 4 dimensions (d1, d2, d3, p)")
    
    d1, d2, d3, p = tensor.shape
    new_shape = (d1 * d2 * d3, p)
    
    reshaped_tensor = tensor.view(new_shape)
    
    return reshaped_tensor


def center_point_cloud_to_unit_cube(point_cloud):
    centroid = point_cloud.mean(dim=0)
    translated_points = point_cloud - centroid
    
    max_abs_val = torch.max(torch.abs(translated_points))
    
    scaled_points = translated_points / (2 * max_abs_val)
    
    centered_points = scaled_points + 0.5
    return centered_points

import torch.nn.functional as F

def calculate_mse_accuracy(input_tensor, output_tensor):
    mse_loss = F.mse_loss(input_tensor, output_tensor, reduction='mean')
    return mse_loss.item()

def calculate_angle_between_tensors(input_tensor, output_tensor):
    # Normalize the input and output tensors to prevent division by zero
    input_tensor_norm = input_tensor / (input_tensor.norm(dim=1, keepdim=True) + 1e-6)
    output_tensor_norm = output_tensor / (output_tensor.norm(dim=1, keepdim=True) + 1e-6)
    
    # Calculate the dot product along the channel dimension
    dot_product = (input_tensor_norm * output_tensor_norm).sum(dim=1)
    
    # Clamp the dot product values to be within the range [-1, 1] to avoid NaN errors in arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians using arccos
    angles = torch.acos(dot_product)
    
    return angles

def calculate_difference_norm(input_tensor, output_tensor):
    # Calculate the difference between the two tensors
    difference = input_tensor - output_tensor
    
    # Calculate the norm of the difference at each point on the grid
    difference_norms = difference.norm(dim=1)
    
    return difference_norms

def load_tensorboard_data(log_dir, tag='val_loss_epoch'):
    print("read metric from ", log_dir)

    # Step 2: Use event_accumulator to load the data
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={ 
                                                event_accumulator.SCALARS: 0,  # 0 means load all scalar events
                                            })

    ea.Reload()  # Loads events from file

    # Example: Extract scalar data
    scalar_tags = ea.Tags()['scalars']  # Get all tags of scalar data logged
    #print(f"Available scalar tags: {scalar_tags}")

    # Assuming you have a scalar tag named 'loss'
    if tag in scalar_tags:
        loss_values = ea.Scalars(tag)  # Get all events for the 'loss' tag
        # Each event is a namedtuple with (wall_time, step, value)
        loss_data = [event.value for event in loss_values]  # Extract loss values

        return loss_data
    else:
        print("Tag not found in scalar tags.")
        
def add_metrics_to_result_file(data_dict):
    file_path = get_project_root() / "results.csv"
    file_exists = os.path.isfile(file_path)  # Check if file already exists
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        
        if not file_exists:
            writer.writeheader()  # Write header only if file didn't exist
        
        writer.writerow(data_dict)  # Write the data row
        
        
def vector_field_to_points(vector_field, cube_size=128, threshold=0):
    vel_points = reshape_tensor(vector_field.permute(1,2,3,0))
    x = torch.arange(cube_size)
    y = torch.arange(cube_size)
    z = torch.arange(cube_size)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    
    cond = torch.norm(vel_points, dim=1) > threshold
    # (vel_points != 0).all(dim=1)

    non_zero_vel_points = vel_points[cond]
    non_zero_mask_points = grid[cond]

    return non_zero_vel_points, non_zero_mask_points


def rotate_vector_field(vector_field, angle, center=[0,0,0]):
    """
    Rotiert ein 3D-Vektorfeld um die Z-Achse um einen gegebenen Punkt.
    
    :param vector_field: Ein Array von Vektoren (x, y, z).
    :param center: Der Punkt (x, y, z), um den rotiert wird.
    :param angle: Der Rotationswinkel in Grad.
    :return: Das rotierte Vektorfeld.
    """
    # Umrechnung des Winkels von Grad in Radiant
    angle_rad = np.radians(angle)
    
    # Rotationsmatrix für die Z-Achse
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])
    
    # Rotiertes Vektorfeld
    rotated_field = np.zeros_like(vector_field)
    
    for i, vector in enumerate(vector_field):

        # Verschiebung zum Ursprung
        shifted_vector = vector - center
        
        # Anwendung der Rotation
        rotated_vector = np.dot(rotation_matrix, shifted_vector)
        
        # Zurückverschiebung
        rotated_field[i] = rotated_vector + center
    
    return rotated_field