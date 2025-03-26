import os
import numpy as np
import h5py

def read_pcd_file(filepath):
    """Read .PCD point cloud file, skipping header"""
    with open(filepath, 'r') as f:
        # Skip header lines
        while True:
            line = f.readline().strip()
            if line.startswith('DATA'):
                break
        
        # Read point cloud data
        data = np.loadtxt(f)
    
    return data[:, :3]  # x, y, z coordinates

def convert_pcd_to_hdf5_cls(input_folder, output_filepath, num_points=1024):
    """
    Convert .PCD files to HDF5 with correct structure for PointNet classification
    """
    point_clouds = []
    labels = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            filepath = os.path.join(input_folder, filename)
            points = read_pcd_file(filepath)
            
            # Sample or pad points to consistent size
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            else:
                points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
            
            point_clouds.append(points)
            # All points are labeled as class 0 (banana)
            labels.append(0)
    
    point_clouds = np.array(point_clouds)
    labels = np.array(labels).reshape(-1, 1)
    
    with h5py.File(output_filepath, 'w') as hf:
        hf.create_dataset('data', data=point_clouds)
        hf.create_dataset('label', data=labels)
        
    print(f"Created H5 file with {len(point_clouds)} point clouds")

# Create directory structure
os.makedirs('data/banana_dataset', exist_ok=True)

# Create train and test datasets
input_folder = './banana_1'
output_train = './data/banana_dataset/train_banana.h5'
output_test = './data/banana_dataset/test_banana.h5'

# Convert all data to H5
convert_pcd_to_hdf5_cls(input_folder, output_train)
# For testing, we'll use the same data
convert_pcd_to_hdf5_cls(input_folder, output_test)

# Create file lists
with open('./data/banana_dataset/train_files.txt', 'w') as f:
    f.write('data/banana_dataset/train_banana.h5\n')

with open('./data/banana_dataset/test_files.txt', 'w') as f:
    f.write('data/banana_dataset/test_banana.h5\n')

print("Conversion complete and file lists created!")
