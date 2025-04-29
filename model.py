import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape, units=50):
    model = Sequential([
        Input(shape=input_shape, dtype=tf.float32, name='input_layer'),
        LSTM(units, return_sequences=True, kernel_initializer='glorot_uniform', 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),  # Increased dropout
        LSTM(units, return_sequences=False, kernel_initializer='glorot_uniform',
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),  # Increased dropout
        Dense(units//2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X, y, epochs=30, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Increased validation split
        callbacks=[early_stop],
        verbose=0
    )
    return model, history

def predict_future(model, last_sequence, future_steps, scaler):
    """Generate future predictions with error handling"""
    if model is None or last_sequence is None:
        return None
    
    try:
        # Ensure proper input shape
        if not isinstance(last_sequence, np.ndarray):
            last_sequence = np.array(last_sequence, dtype=np.float32)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(future_steps):
            # Reshape for model input
            x_input = current_sequence.reshape((1, len(current_sequence), 1))
            
            # Make prediction
            next_pred = model.predict(x_input, verbose=0)[0,0]
            future_predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # Convert and scale predictions
        return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    except Exception as e:
        print(f"Prediction error: {e}")
        return None