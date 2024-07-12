import numpy as np
from pathlib import Path
import torch
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay

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

    ranges = points_max - points_min
    if fix_resizing_factor:
        max_range = ranges.max()
    else:
        max_range = (data_points - points_min) / (points_max - points_min)
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