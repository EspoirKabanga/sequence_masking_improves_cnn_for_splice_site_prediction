'''
DeepSplicer model architecture

'''

import tensorflow as tf

def DeepSplicer(length):
    """
    Defines the DeepSplicer model architecture for splice site prediction.

    Args:
    - length (int): Length of the input DNA sequence.

    Returns:
    - model: A compiled Keras Sequential model.
    """
    model = tf.keras.models.Sequential(name='DeepSplicer')

    # Adding three Conv1D layers
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', 
                                     batch_input_shape=(None, length, 4), activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Dense layer with 100 neurons and ReLU activation
    model.add(tf.keras.layers.Dense(100, activation='relu'))

    # Dropout layer to prevent overfitting
    model.add(tf.keras.layers.Dropout(0.3))

    # Final dense layer with softmax activation for classification
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # Compile the model with Adam optimizer and categorical crossentropy loss
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
