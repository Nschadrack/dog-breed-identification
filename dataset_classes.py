import numpy as np

import warnings
import os
from PIL import Image
import collections
import shutil
import math

warnings.simplefilter('ignore')


def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
        
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def reorg_dog_data(data_dir, labels, valid_ratio):
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


if __name__ == "__main__":
    ROOT_DATA_DIR = 'data'
    TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'test')
    TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'train')

    valid_ratio = 0.2

    labels = read_csv_labels(os.path.join(ROOT_DATA_DIR, 'labels.csv'))
    reorg_dog_data(ROOT_DATA_DIR, labels, valid_ratio)
    
