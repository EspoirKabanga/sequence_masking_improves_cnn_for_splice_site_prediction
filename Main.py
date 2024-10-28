'''
This script trains the DeepSplicer model on a dataset with splice site masking strategies. 
It performs 5-fold cross-validation using a masked dataset (acceptor splice sites) and includes callbacks for early stopping and saving the best model weights based on validation loss. 
The model is trained on various masking strategies, such as upstream, downstream, and bidirectional masking. 
After each fold, the best model is saved, and GPU memory is cleared to optimize performance.

'''

import os
import tensorflow as tf
import numpy as np
import random
from keras import utils, backend as K
from sklearn.model_selection import StratifiedKFold
import glob
import shutil
from read_all_data import read_and_split_data
import CNN_models as mdl
import callbacks as clbk

# Set environment variables for TensorFlow and CUDA configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# Set seed for reproducibility
seed_value = 42
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Data paths
pos_data = f'RDM_dataset_final/random_SPC_arab_acceptors_final.pos'
neg_data = f'RDM_dataset_final/random_SPC_arab_acceptors_final.neg'

# Masking configurations
mask_sizes = [i * 15 for i in range(0, 1)]  # Example mask sizes, adjust for your experiments
mask_position = ['bidirectional']  # Masking strategies: 'bidirectional', 'upstream', 'downstream'

# Loop over different masking sizes
for size in mask_sizes:
    for pos in mask_position:

        # Read and preprocess data
        train_X, train_y = read_and_split_data(pos_data, neg_data, size, pos, 200, 200, 'train')
        seq_length = len(train_X[0])

        # Define path for saving results
        pth = f'RDM_SPC_acc_{pos}_{size}'
        if not os.path.exists(f"/home/ekabanga/BIBM_masking/{pth}"):
            os.makedirs(f"/home/ekabanga/BIBM_masking/{pth}")

        # Initialize the model
        model = mdl.DeepSplicer(seq_length)

        # KFold Cross Validation with 5 splits
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold = 1

        for train_index, valid_index in kf.split(train_X, train_y):
            X_train, X_valid = train_X[train_index], train_X[valid_index]
            y_train, y_valid = train_y[train_index], train_y[valid_index]

            print(f'Size of train = {len(X_train)}')
            print(f'Size of valid = {len(X_valid)}')
            print('--------------------------------------')

            # Reshape and prepare data for model input
            X_train = X_train.reshape(-1, seq_length, 4).astype(np.float32)
            X_valid = X_valid.reshape(-1, seq_length, 4).astype(np.float32)
            y_train = utils.to_categorical(y_train, num_classes=2).astype(np.float32)
            y_valid = utils.to_categorical(y_valid, num_classes=2).astype(np.float32)

            # Create callbacks for model checkpoint and early stopping
            model_callbacks = clbk.my_callbacks(pth, f'RDM_SPC_acc_{pos}_{size}_{model.name}_fold_{fold}')

            print(f"Fold {fold} for SPC_acc_mask_size{size}:\n")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples (from fold): {len(X_valid)}")

            # Train the model
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=64, callbacks=model_callbacks, verbose=1)

            # Load the best model weights based on validation loss
            list_of_files = glob.glob(f'{pth}/*')
            latest_file = max(list_of_files, key=os.path.getctime)
            model.load_weights(latest_file)

            # Save the model
            save_dir = f"/home/ekabanga/BIBM_masking/RDM_saved_models/RDM_SPC_acc_{pos}_{size}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save(f'{save_dir}/RDM_SPC_acc_{pos}_{size}_{model.name}_fold_{fold}.h5')

            fold += 1

            # Clear session to free up GPU memory
            K.clear_session()

            # Remove model best weight directory
            shutil.rmtree(f'{pth}')
