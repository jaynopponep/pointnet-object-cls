import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name [default: pointnet_cls]')
parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--input_file', required=True, help='Input .pcd file to classify')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes [default: 2]')
FLAGS = parser.parse_args()

# Set constants
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_CLASSES = FLAGS.num_classes

# Import model
MODEL = importlib.import_module(FLAGS.model)
MODEL.NUM_CLASSES = NUM_CLASSES

# Class labels - update with your actual class names
CLASS_NAMES = ['banana', 'orange']  # Modify as needed for your classes

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

def preprocess_point_cloud(points, num_points=1024):
    """Preprocess point cloud to have consistent number of points"""
    if len(points) > num_points:
        # Randomly sample points
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    else:
        # Pad with zeros if not enough points
        points = np.pad(points, ((0, num_points - len(points)), (0, 0)), mode='constant')
    
    # Normalize to unit sphere (optional but recommended)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / furthest_distance
    
    return points

def classify():
    """Classify a point cloud file"""
    # Load and preprocess the point cloud
    print(f"Reading point cloud from {FLAGS.input_file}")
    points = read_pcd_file(FLAGS.input_file)
    processed_points = preprocess_point_cloud(points, NUM_POINT)
    
    # Add batch dimension (batch size of 1)
    processed_points = np.expand_dims(processed_points, 0)
    
    # Create TensorFlow session and load model
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX) if GPU_INDEX >= 0 else '/cpu:0'):
            # Define placeholders for the model
            pointclouds_pl, _ = MODEL.placeholder_inputs(1, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Get model and predictions
            pred, _ = MODEL.get_model(pointclouds_pl, is_training_pl)
            pred_softmax = tf.nn.softmax(pred)
            
            # Create saver for restoring model
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        # Restore model
        saver.restore(sess, MODEL_PATH)
        print(f"Model restored from: {MODEL_PATH}")
        
        # Run inference
        feed_dict = {
            pointclouds_pl: processed_points,
            is_training_pl: False
        }
        
        pred_val = sess.run(pred_softmax, feed_dict=feed_dict)
        pred_label = np.argmax(pred_val[0])
        confidence = pred_val[0][pred_label]
        
        # Print results
        print(f"\nClassification result:")
        print(f"  Predicted class: {CLASS_NAMES[pred_label]} (Class ID: {pred_label})")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Print all class probabilities
        print("\nClass probabilities:")
        for i, prob in enumerate(pred_val[0]):
            print(f"  {CLASS_NAMES[i]}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == "__main__":
    classify()
