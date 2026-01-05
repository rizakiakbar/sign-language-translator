import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import joblib  # ✅ TAMBAHKAN INI
from src.config import Config

class CoordinateModel:
    def __init__(self):
        self.config = Config()
        self.model = None
        
    def build_model(self):
        """Bangun model neural network untuk koordinat"""
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.config.NUM_FEATURES,)),
            
            # Hidden layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, data_loader=None):
        """Training model"""
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # ✅ SAVE SCALER setelah training
        if data_loader is not None:
            scaler_path = os.path.join(self.config.MODELS_DIR, 'scaler.pkl')
            joblib.dump(data_loader.scaler, scaler_path)
            print(f"✅ Scaler saved to: {scaler_path}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        if self.model is None:
            raise ValueError("Model belum di-training")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy