'''
This script defines a function that generates a set of TensorFlow callbacks for model training. 
It includes a model checkpoint callback to save the best model based on validation loss, and an early stopping callback to halt training when validation loss stops improving. 
These callbacks help in saving the best model and preventing overfitting by stopping training at the optimal point.

'''

import tensorflow as tf
import os

def my_callbacks(path, project_name=None):
    """
    Creates and returns a list of callbacks for model training.

    Args:
    - path (str): Directory path where model weights will be saved.
    - project_name (str, optional): Name of the project, used to name the saved weight files.

    Returns:
    - list: List of TensorFlow callbacks (ModelCheckpoint, EarlyStopping).
    """

    # Create the checkpoint file path
    checkpoint_filepath = os.path.join(path, f'{project_name}_weights.{{epoch:02d}}-{{val_loss:.2f}}.h5')

    # Callback for saving the model's best weights based on validation loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_freq='epoch',
        save_best_only=True
    )

    # Early stopping to stop training when validation loss stops improving
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=10,
        restore_best_weights=True
    )

    # List of callbacks to use during model training
    my_callback = [model_checkpoint_callback, early_stopping_callback]

    return my_callback
