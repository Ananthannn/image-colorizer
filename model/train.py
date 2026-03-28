import os
import tensorflow as tf
from architecture import build_colorization_model

def train():
    # Hyperparameters
    batch_size = 16
    epochs = 50
    learning_rate = 1e-3
    checkpoint_dir = "../checkpoints"

    # Build and compile model
    model = build_colorization_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Load dataset
    # TODO: Replace with actual data loading logic
    # Note: Ensure target A/B channels are divided by 128 to match tanh range [-1, 1]
    # X_train, Y_train = load_data(...)
    
    # Setup checkpoints
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "colorizer_weights.keras"),
        monitor="loss", 
        save_best_only=True,
        verbose=1
    )
    
    # Train
    '''
    model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    '''

if __name__ == "__main__":
    train()
