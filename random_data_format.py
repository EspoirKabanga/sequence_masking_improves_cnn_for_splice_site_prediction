import os
import random
import sys

# Set CUDA environment
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Seed for reproducibility
random.seed(42)

def read_inputs(fname):
    """
    Reads input sequences from a file and returns them as a list.
    """
    with open(fname) as file:
        return [line.strip() for line in file.readlines()]

def replace_nucleotides(sequence, up_length, down_length, position):
    """
    Replaces nucleotides with 'N' at the upstream or downstream ends of the sequence.
    
    Args:
    - sequence (str): The input DNA sequence.
    - up_length (int): Length of upstream region.
    - down_length (int): Length of downstream region.
    - position (str): Either 'upstream' or 'downstream' indicating which end to mask.
    
    Returns:
    - str: The modified sequence with nucleotides replaced by 'N'.
    """
    final_sequence = list(sequence)
    
    up_replacements = random.randint(1, round(up_length * 0.1))
    down_replacements = random.randint(1, round(down_length * 0.1))
    
    if position == 'upstream':
        final_sequence[:up_replacements] = ['N'] * up_replacements
    elif position == 'downstream':
        final_sequence[-down_replacements:] = ['N'] * down_replacements
    
    return ''.join(final_sequence)

def process_sequence(sequence, seq_length):
    """
    Removes 'N' nucleotides from a sequence, then pads it with 'N' to a target length.
    
    Args:
    - sequence (str): The input DNA sequence.
    - seq_length (int): Desired length for padding.
    
    Returns:
    - str: The sequence padded with 'N' to the specified length.
    """
    
    # Trimming
    sequence = sequence.replace('N', '')
    total_nucleotides = len(sequence)
    num_nucleotides_to_add = seq_length - total_nucleotides
    num_nucleotides_to_add_half = num_nucleotides_to_add // 2

    beginning_n = random.randint(0, num_nucleotides_to_add_half)
    end_n = num_nucleotides_to_add - beginning_n

    final_sequence = 'N' * beginning_n + sequence + 'N' * end_n

    return final_sequence

# Example sequences
chrom = [1]
data_file = f'New_DataSet/DRANetSplicer/oryza_acceptor_negative.txt'

sequences = read_inputs(data_file)
print(f"Number of sequences: {len(sequences)}")

# Replace 'N' in the upstream for the first half and in the downstream for the second half
for i in range(len(sequences) // 2):
    sequences[i] = replace_nucleotides(sequences[i], 200, 200, 'upstream')

for i in range(len(sequences) // 2, len(sequences)):
    sequences[i] = replace_nucleotides(sequences[i], 200, 200, 'downstream')

# Write modified sequences to file
output_file = 'RDM_dataset_count/random_DRANet_acceptors.neg'
with open(output_file, 'w') as f:
    sys.stdout = f
    for sequence in sequences:
        print(sequence)
sys.stdout = sys.__stdout__

# Count 'N' nucleotides in each sequence and write results to file
counts_list = [{'sequence': seq, 'count': seq.count('N')} for seq in sequences]

count_output_file = 'RDM_dataset_count/random_DRANet_acceptors_count.neg'
with open(count_output_file, 'w') as f:
    sys.stdout = f
    for counts in counts_list:
        print(f"Sequence: {counts['sequence']}, Count: {counts['count']}")
sys.stdout = sys.__stdout__

# Process and pad sequences to the target length
processed_sequences = [process_sequence(seq, 402) for seq in sequences]

# Write the processed sequences to the final file
final_output_file = 'RDM_dataset_final/random_DRANet_acceptors_final.neg'
with open(final_output_file, 'w') as f:
    sys.stdout = f
    for sequence in processed_sequences:
        print(sequence)
sys.stdout = sys.__stdout__
