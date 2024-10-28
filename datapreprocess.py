'''
This script contains utility functions for applying sequence masking and one-hot encoding DNA sequences. 
The replace_nucleotides and replace_both functions enable the masking of upstream, downstream, or bidirectional regions.
The one_hot_encode function converts DNA sequences into a one-hot encoded format. 
The read_file function reads sequences from input files, applies masking, and removes duplicates.

''' 

import numpy as np
import random
from collections import OrderedDict

# Set seed for reproducibility
seed_value = 42
random.seed(seed_value)

def replace_nucleotides(sequence, num_replacements, position):
    """
    Replace nucleotides in a DNA sequence with 'N' based on the masking position.
    
    Args:
    - sequence (str): The DNA sequence.
    - num_replacements (int): The number of nucleotides to replace.
    - position (str): Position of masking ('upstream', 'downstream', 'bidirectional').

    Returns:
    - str: The masked sequence.
    """
    if len(sequence) == 0:
        return sequence

    if num_replacements >= len(sequence):
        raise ValueError("Number of replacements must be less than the sequence length.")
    
    if position not in ['upstream', 'downstream', 'bidirectional']:
        raise ValueError("Position must be 'upstream', 'downstream', or 'bidirectional'.")

    final_sequence = list(sequence)

    if position == 'upstream':
        final_sequence[:num_replacements] = ['N'] * num_replacements
    elif position == 'downstream':
        final_sequence[-num_replacements:] = ['N'] * num_replacements
    elif position == 'bidirectional':
        final_sequence[:num_replacements] = ['N'] * num_replacements
        final_sequence[-num_replacements:] = ['N'] * num_replacements

    return ''.join(final_sequence)

def replace_both(sequence, replace_upstream, replace_downstream):
    """
    Replace nucleotides both upstream and downstream in a sequence.
    
    Args:
    - sequence (str): The DNA sequence.
    - replace_upstream (int): Number of nucleotides to mask in the upstream.
    - replace_downstream (int): Number of nucleotides to mask in the downstream.
    
    Returns:
    - str: The masked sequence.
    """
    if len(sequence) == 0:
        return sequence

    final_sequence = list(sequence)

    # Apply masking to both ends
    final_sequence[:replace_upstream] = ['N'] * replace_upstream
    final_sequence[-replace_downstream:] = ['N'] * replace_downstream

    return ''.join(final_sequence)

def one_hot_encode(sequences):
    """
    One-hot encodes a list of DNA sequences.
    
    Args:
    - sequences (list): List of DNA sequences.
    
    Returns:
    - list: One-hot encoded sequences.
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # Masked regions
    }

    return [[mapping[nuc] for nuc in seq] for seq in sequences]

def read_file(f1, f2, size, pos, up_length, down_length):
    """
    Reads DNA sequences from files and applies specified masking strategies.
    
    Args:
    - f1 (str): File containing positive (splice site) sequences.
    - f2 (str): File containing negative (non-splice site) sequences.
    - size (int): Percentage of the sequence to mask.
    - pos (str): Masking position ('upstream', 'downstream', 'bidirectional').
    - up_length (int): Length of the upstream region.
    - down_length (int): Length of the downstream region.

    Returns:
    - tuple: One-hot encoded positive and negative sequences.
    """
    print(f"\nProcessing sequences with {size}% {pos} masking...\n")

    # Calculate masking sizes
    if pos == 'upstream':
        size = round((up_length * size) / 100)
    elif pos == 'downstream':
        size = round((down_length * size) / 100)
    elif pos == 'bidirectional':
        size_up = round((up_length * size) / 100)
        size_down = round((down_length * size) / 100)

    # Read sequences from the files
    lines_pos = open(f1).readlines()
    lines_neg = open(f2).readlines()

    pos_lst = [seq.strip() for seq in lines_pos if len(seq.strip()) > 0]
    neg_lst = [seq.strip() for seq in lines_neg if len(seq.strip()) > 0]

    mask_pos, mask_neg = [], []

    # Apply masking
    if pos == 'upstream' or pos == 'downstream':
        mask_pos = [replace_nucleotides(seq, size, pos) for seq in pos_lst]
        mask_neg = [replace_nucleotides(seq, size, pos) for seq in neg_lst]
    elif pos == 'bidirectional':
        mask_pos = [replace_both(seq, size_up, size_down) for seq in pos_lst]
        mask_neg = [replace_both(seq, size_up, size_down) for seq in neg_lst]

    # Remove duplicates
    mask_pos = list(OrderedDict.fromkeys(mask_pos))
    mask_neg = list(OrderedDict.fromkeys(mask_neg))

    # Example of masked sequence
    print(mask_pos[0])

    return one_hot_encode(mask_pos), one_hot_encode(mask_neg)
