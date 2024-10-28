'''
This script handles the preprocessing of DNA sequences by applying masking strategies to upstream, downstream, or both regions. 
It also reads input files, one-hot encodes sequences, and splits the data into training and test sets. 
The replace_nucleotides and replace_both functions allow flexible masking strategies for analyzing splice site sequences.

'''

import numpy as np
import random
from sklearn.model_selection import train_test_split
import datapreprocess as prep

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

def read_and_split_data(pos_data, neg_data, size, pos, up_length, down_length, partition, subset_size=None):  
    """
    Reads and processes the data, applying masking strategies and splitting it into training or test sets.

    Parameters:
    ----------
    pos_data : str
        File path containing DNA sequences with true splice sites.
    neg_data : str
        File path containing DNA sequences without true splice sites.
    size : int
        Masking size (e.g., 0 for unmasked, 15, 30, 45, 60, 75, or 90%).
    pos : str
        Masking position ('upstream', 'downstream', 'bidirectional').
    up_length : int
        Length of the upstream region.
    down_length : int
        Length of the downstream region.
    partition : str
        Specify 'train' or 'test' to indicate data partitioning.
    subset_size : int, optional
        Subset size for datasets like DRANnetSplicer where sampling is necessary.

    Returns:
    --------
    tuple:
        X (np.array), y (np.array) corresponding to the input sequences and their labels (1 for positive, 0 for negative).
    """
    print(f'\nProcessing data with {size}% masking, {pos} position...\n')

    # Reading the positive and negative sequence files
    data_pos, data_neg = prep.read_file(pos_data, neg_data, size, pos, up_length, down_length)

    # Optional: sample subsets for certain datasets (like DRANnetSplicer)
    if subset_size is not None:
        data_pos = random.sample(data_pos, subset_size)
        data_neg = random.sample(data_neg, subset_size)

    # Combine positive and negative datasets, and create corresponding labels
    Xd = data_pos + data_neg
    yd = [1] * len(data_pos) + [0] * len(data_neg)

    # Splitting data into training and test sets
    train_X, test_X, train_y, test_y = train_test_split(np.array(Xd), np.array(yd), test_size=0.2, random_state=42, stratify=yd)

    # Return the appropriate partition
    if partition == 'train':
        print(f'Length of training set = {len(train_X)}')
        return train_X, train_y
    elif partition == 'test':
        print(f'Length of test set = {len(test_X)}')
        return test_X, test_y
    else:
        raise ValueError("Partition must be 'train' or 'test'.")
