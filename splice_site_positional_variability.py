'''
This script introduces positional variability to splice sites by splitting the dataset, trimming sequences from either the upstream or downstream end based on the defined lengths, and padding the sequences with 'N' nucleotides to maintain their original length.
The trimmed length is random number from 0 to 10% of either the upstream length or the downstream length.

'''

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import sys

random.seed(42) # It is recommended to use a differnt seed value for positive data and negative data if the dataset is balanced

def readInputs(fname):
    lines_pos = open(fname).readlines()

    pos_list = []

    for line in lines_pos:
        pos_list.append(line.strip())

    return pos_list

def replace_nucleotides(sequence, up_length, down_length, position):

    final_sequence = list(sequence)

    up_replacements = random.randint(1, round(up_length * 0.1))
    down_replacements = random.randint(1, round(down_length * 0.1))

    if position=='upstream':
        final_sequence[:up_replacements] = ['N'] * up_replacements
    
    if position=='downstream':
        final_sequence[-down_replacements:] = ['N'] * down_replacements

    return ''.join(final_sequence)

def process_sequence(sequence, seq_length):
    # Remove all 'N' from the sequence
    sequence = sequence.replace('N', '')

    # Calculate the number of 'U' nucleotides to add at the beginning and end
    total_nucleotides = len(sequence)
    num_nucleotides_to_add = seq_length - total_nucleotides
    num_nucleotides_to_add_half = num_nucleotides_to_add // 2

    # Add random number of 'U' nucleotides at the beginning and end
    beginning_u = random.randint(0, num_nucleotides_to_add_half)
    end_u = num_nucleotides_to_add - beginning_u

    # Generate the final sequence
    final_sequence = 'N' * beginning_u + sequence + 'N' * end_u

    return final_sequence


# Example sequences

# chrom = [1]

data_file = f'New_DataSet/DRANetSplicer/oryza_acceptor_negative.txt'

sequences = readInputs(data_file)

print(len(sequences))

# import pdb
# pdb.set_trace()

# Task 1: Replace 'n' nucleotides at the beginning for the first half sequences
for i in range(len(sequences)//2):
    sequences[i] = replace_nucleotides(sequences[i], 200, 200, 'upstream')

# Task 2: Replace 'n' nucleotides at the end for the last half sequences
for i in range(len(sequences)//2, len(sequences)):
    sequences[i] = replace_nucleotides(sequences[i], 200, 200, 'downstream')

with open(f'RDM_dataset_count/random_DRANet_acceptors.neg', 'w') as f:
    sys.stdout = f

    # Print the modified sequences
    for sequence in sequences:
        print(sequence)

# Reset standard output
sys.stdout = sys.__stdout__


counts_list = []
for sequence in sequences:
    count = sequence.count('N')
    counts_list.append({'sequence': sequence, 'count': count})

with open(f'RDM_dataset_count/random_DRANet_acceptors_count.neg', 'w') as f:
    sys.stdout = f

    # Print the counts
    for counts in counts_list:
        print(f"Sequence: {counts['sequence']}, Count: {counts['count']}")

# Reset standard output
sys.stdout = sys.__stdout__


# Process each sequence
processed_sequences = []
for sequence in sequences:
    processed_sequence = process_sequence(sequence, 402)
    processed_sequences.append(processed_sequence)


with open(f'RDM_dataset_final/random_DRANet_acceptors_final.neg', 'w') as f:
    sys.stdout = f

    # Print the processed sequences
    for sequence in processed_sequences:
        print(sequence)

# Reset standard output
sys.stdout = sys.__stdout__

