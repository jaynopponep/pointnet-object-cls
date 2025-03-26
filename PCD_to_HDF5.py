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

def convert_pcd_to_hdf5_cls(class_folders, output_filepath, num_points=1024):
    """
    Convert .PCD files from multiple class folders to HDF5 with correct structure for PointNet classification
    
    Args:
        class_folders: Dictionary mapping class names to folder paths
        output_filepath: Path to save the output HDF5 file
        num_points: Number of points per point cloud
    """
    point_clouds = []
    labels = []
    
    # Dictionary to track statistics
    stats = {class_name: 0 for class_name in class_folders.keys()}
    
    # Process each class
    for class_idx, (class_name, folder_path) in enumerate(class_folders.items()):
        print(f"Processing class {class_name} (label: {class_idx}) from {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.pcd'):
                filepath = os.path.join(folder_path, filename)
                points = read_pcd_file(filepath)
                
                # Sample or pad points to consistent size
                if len(points) > num_points:
                    indices = np.random.choice(len(points), num_points, replace=False)
                    points = points[indices]
                else:
                    points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
                
                point_clouds.append(points)
                # Assign label based on class index
                labels.append(class_idx)
                stats[class_name] += 1
    
    point_clouds = np.array(point_clouds)
    labels = np.array(labels).reshape(-1, 1)
    
    with h5py.File(output_filepath, 'w') as hf:
        hf.create_dataset('data', data=point_clouds)
        hf.create_dataset('label', data=labels)
    
    print(f"Created H5 file {output_filepath} with {len(point_clouds)} point clouds")
    print("Class distribution:")
    for class_name, count in stats.items():
        print(f"  - {class_name}: {count} samples")
    
    return stats

# Create directory structure
os.makedirs('data/dataset', exist_ok=True)

# Define class folders
# IMPORTANT: Add all your class folders here
class_folders = {
    'banana': './trainingdata/banana_1',
    'orange': './trainingdata/orange'  # Add your orange folder path here
}

# Create train and test datasets
output_train = './data/dataset/train.h5'
output_test = './data/dataset/test.h5'

# Convert all data to H5 for training
stats = convert_pcd_to_hdf5_cls(class_folders, output_train)

# For testing, we'll use the same data
# In a real application, you might want to split your data into train/test
convert_pcd_to_hdf5_cls(class_folders, output_test)

# Create file lists
with open('./data/dataset/train_files.txt', 'w') as f:
    f.write('data/dataset/train.h5\n')

with open('./data/dataset/test_files.txt', 'w') as f:
    f.write('data/dataset/test.h5\n')

print("Conversion complete and file lists created!")
