import numpy as np


for i in range(10000):
    # Generate random point clouds
    num_points = 100000
    num_dimensions = 3
    point_clouds = np.random.rand(num_points, num_dimensions)

    # Center the point clouds
    centered_point_clouds = point_clouds - np.mean(point_clouds, axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_point_clouds, rowvar=False)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Get the principal components
    principal_components = sorted_eigenvectors[:, :num_dimensions]

    # Print the principal components
    print("Principal Components:")
    print(principal_components)