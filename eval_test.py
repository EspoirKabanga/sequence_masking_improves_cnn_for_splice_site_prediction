'''
This script evaluates the DeepSplicer model on masked datasets for DRANet Oryza acceptor sequences. 
It uses different masking strategies (e.g., bidirectional, upstream, downstream) and iterates through mask sizes (15%, 30%, etc.). 
The model is evaluated on multiple metrics, including Precision, Recall, F1-score, MCC, FPR80, PR95, auPRC, and auROC. 
Results are saved into CSV files.

'''

import os
import tensorflow as tf
import numpy as np
import random
import keras
from keras import utils
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef
import csv
from read_all_data import read_and_split_data

# Set environment variables for TensorFlow and CUDA configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Set seed for reproducibility
seed_value = 42
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Data paths
pos_data = f'RDM_dataset_final/random_DRANet_acceptors_final.pos'
neg_data = f'RDM_dataset_final/random_DRANet_acceptors_final.neg'

# Masking strategies
masking_strategies = ['bidirectional']  # Possible values: 'bidirectional', 'upstream', 'downstream'

# Iterate over each masking strategy
for strategy in masking_strategies:
    # Set start index for mask sizes based on strategy
    start_index = 0 if strategy == 'bidirectional' else 1

    # Define mask sizes for this strategy (adjust based on RDM data specifics)
    mask_sizes = [i * 15 for i in range(start_index, 1)]  # Modify range for non-RDM datasets if needed

    for mask_size in mask_sizes:
        result_file = f"TFTR/RDM_DRANet_oryza_acceptors_{strategy}_{mask_size}.csv"
        
        # Initialize the CSV file with headers
        with open(result_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Mask size", "Fold", "Precision", "Recall", "F1-score", "MCC", "FPR80", "PR95", "auPRC", "auROC"])

        # Read and preprocess the test data
        test_X, test_y = read_and_split_data(pos_data, neg_data, mask_size, strategy, 200, 200, 'test')
        seq_len = len(test_X[0])
        test_X = test_X.reshape(-1, seq_len, 4)
        test_y = utils.to_categorical(test_y, num_classes=2)

        # Iterate over folds
        for fold in range(1, 6):
            # Load the pre-trained model for this fold
            model_path = f'saved_models/DRANet_oryza_acc_{strategy}_{mask_size}/DRANet_oryza_acc_{strategy}_{mask_size}_DeepSplicer_fold_{fold}.h5'
            model = keras.models.load_model(model_path)

            # Evaluate model and generate predictions
            loss, accuracy = model.evaluate(test_X, test_y)
            pred = model.predict(test_X)

            # Convert predictions to binary with a threshold of 0.5
            threshold = 0.5
            binary_pred = (pred > threshold).astype(int)

            # Calculate evaluation metrics
            precision_scores = precision_score(test_y, binary_pred, average=None)
            recall_scores = recall_score(test_y, binary_pred, average=None)
            f1_scores = f1_score(test_y, binary_pred, average=None)
            mcc = matthews_corrcoef(test_y.argmax(axis=1), binary_pred.argmax(axis=1))

            # Calculate FPR80
            fpr, tpr, roc_thresholds = roc_curve(test_y.ravel(), pred.ravel())
            fpr80 = fpr[np.argmax(tpr >= 0.8)]
            
            # Calculate PR95
            precision, recall, pr_thresholds = precision_recall_curve(test_y.ravel(), pred.ravel())
            pr95_index = np.argmax(recall >= 0.95)
            pr95 = precision[pr95_index] if pr95_index < len(precision) else np.nan

            # Calculate auPRC and auROC
            auprc = auc(recall, precision)
            auroc = auc(fpr, tpr)

            # Log fold-specific results
            print(f"Mask size {mask_size}, Fold {fold} - FPR80: {fpr80}, PR95: {pr95}, auPRC: {auprc}, auROC: {auroc}")

            # Write fold results to the CSV file
            with open(result_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([mask_size, fold, round(precision_scores[1], 4), round(recall_scores[1], 4), 
                                 round(f1_scores[1], 4), round(mcc, 4), round(fpr80, 4), round(pr95, 4), 
                                 round(auprc, 4), round(auroc, 4)])

    print("Results have been successfully written to the CSV files.")
