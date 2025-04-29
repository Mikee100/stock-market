import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error

class LSTMForecaster:
    def __init__(self, input_shape, units=64):
        self.input_shape = input_shape
        self.units = units
        self.model = self.build_advanced_model()  # Changed from _build_advanced_model to build_advanced_model
        self.scaler = None
        self.X_train = None
        self.y_train = None
        
    def build_advanced_model(self):  # Removed underscore prefix
        """Build a more sophisticated LSTM model architecture"""
        model = Sequential([
            Input(shape=self.input_shape, dtype=tf.float32, name='input_layer'),
            
            # First LSTM layer with return sequences for deep learning
            LSTM(self.units * 2, return_sequences=True,
                kernel_initializer='he_normal',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.4),
            
            # Second LSTM layer
            LSTM(self.units, return_sequences=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers with regularization
            Dense(self.units, activation='relu',  # Changed from 'swish' to 'relu' for wider compatibility
                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='linear')
        ])
        
        # Custom learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
        
        optimizer = Adam(learning_rate=lr_schedule, clipvalue=0.5)
        
        # Using MAE as loss function for better compatibility
        model.compile(optimizer=optimizer,
                     loss='mean_absolute_error',
                     metrics=['mae', 'mse'])
        
        return model
    
    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.15):
        """Enhanced training with callbacks and validation"""
        self.X_train = X
        self.y_train = y
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        return history
    
    def predict_sequences(self, input_sequence, n_steps):
        """Make multi-step predictions"""
        predictions = []
        current_sequence = input_sequence.copy()
        
        for _ in range(n_steps):
            # Reshape and predict
            x_input = current_sequence.reshape((1, len(current_sequence), 1))
            pred = self.model.predict(x_input, verbose=0)[0,0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        return np.array(predictions)
    
    def predict_with_uncertainty(self, input_sequence, n_steps, n_samples=100):
        """Monte Carlo Dropout for uncertainty estimation"""
        # Enable dropout at test time
        self.model.trainable = True
        
        predictions = []
        for _ in range(n_samples):
            preds = self.predict_sequences(input_sequence, n_steps)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(np.mean((y_test - y_pred)**2)),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Directional accuracy
        direction_true = np.sign(np.diff(y_test.flatten()))
        direction_pred = np.sign(np.diff(y_pred.flatten()))
        metrics['Direction_Accuracy'] = np.mean(direction_true == direction_pred)
        
        return metrics
    
    def save_model(self, filepath):
        """Save model and scaler"""
        self.model.save(filepath)
        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, filepath + '_scaler.pkl')
    
    @classmethod
    def load_model(cls, filepath):
        """Load saved model"""
        import joblib
        model = tf.keras.models.load_model(filepath)
        scaler = joblib.load(filepath + '_scaler.pkl')
        
        forecaster = cls(model.input_shape[1:])
        forecaster.model = model
        forecaster.scaler = scaler
        return forecaster