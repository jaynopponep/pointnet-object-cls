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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name [default: pointnet_cls]')
parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size [default: 32]')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes [default: 2]')
parser.add_argument('--data_dir', default='data/dataset', help='Data directory [default: data/dataset]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_CLASSES = FLAGS.num_classes
DATA_DIR = FLAGS.data_dir

MODEL = importlib.import_module(FLAGS.model)
MODEL.NUM_CLASSES = NUM_CLASSES

TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, DATA_DIR, 'test_files.txt'))

CLASS_NAMES = ['banana', 'orange'] 

def evaluate():
    """Evaluate model on test data"""
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX) if GPU_INDEX >= 0 else '/cpu:0'):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pred, _ = MODEL.get_model(pointclouds_pl, is_training_pl)
            pred_softmax = tf.nn.softmax(pred)
            
            saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        saver.restore(sess, MODEL_PATH)
        print(f"Model restored from: {MODEL_PATH}")
        
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        total_correct = 0
        total_seen = 0
        for fn in range(len(TEST_FILES)):
            print(f"Testing file {TEST_FILES[fn]}")
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:NUM_POINT, :]
            current_label = np.squeeze(current_label)
            
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                
                feed_dict = {
                    pointclouds_pl: current_data[start_idx:end_idx, :, :],
                    labels_pl: current_label[start_idx:end_idx],
                    is_training_pl: False
                }
                
                pred_val = sess.run(pred_softmax, feed_dict=feed_dict)
                pred_labels = np.argmax(pred_val, 1)
                
                for i in range(len(pred_labels)):
                    true_label = int(current_label[start_idx+i])
                    pred_label = int(pred_labels[i])
                    confusion_matrix[true_label, pred_label] += 1
                
                correct = np.sum(pred_labels == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += BATCH_SIZE
                
        overall_accuracy = total_correct / float(total_seen)
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        
        print("Per-class accuracy:")
        class_accuracies = []
        for i in range(NUM_CLASSES):
            class_total = np.sum(confusion_matrix[i, :])
            if class_total > 0:
                class_acc = confusion_matrix[i, i] / class_total
                class_accuracies.append(class_acc)
                print(f"  {CLASS_NAMES[i]}: {class_acc:.4f} ({int(confusion_matrix[i, i])}/{int(class_total)})")
            else:
                print(f"  {CLASS_NAMES[i]}: N/A (no samples)")
        
        mean_class_accuracy = np.mean(class_accuracies)
        print(f"Mean class accuracy: {mean_class_accuracy:.4f}")
        
        print("\nConfusion Matrix:")
        print("  " + " ".join([f"{name[:6]:>6}" for name in CLASS_NAMES]))
        for i in range(NUM_CLASSES):
            row_str = f"{CLASS_NAMES[i][:6]:<6}"
            for j in range(NUM_CLASSES):
                row_str += f"{int(confusion_matrix[i, j]):>6}"
            print(row_str)

if __name__ == "__main__":
    evaluate()
