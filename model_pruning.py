# model_pruning.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from src.config import Config

def prune_model():
    """Prune model untuk mengurangi size"""
    config = Config()
    
    # Load model
    model_path = os.path.join(config.MODELS_DIR, 'coordinate_model.h5')
    model = keras.models.load_model(model_path)
    
    # Pruning configuration
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=0.5,
            begin_step=0,
            frequency=100
        )
    }
    
    try:
        import tensorflow_model_optimization as tfmot
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        
        # Compile pruned model
        model_for_pruning.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for pruning (fine-tuning)
        from src.data_loader import CoordinateDataLoader
        data_loader = CoordinateDataLoader()
        X_train, y_train, X_val, y_val, _, _, _ = data_loader.load_dataset()
        
        # Callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        # Fine-tune dengan pruning
        model_for_pruning.fit(
            X_train, y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Strip pruning wrappers
        final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        # Save pruned model
        pruned_path = os.path.join(config.MODELS_DIR, 'model_pruned.h5')
        final_model.save(pruned_path)
        
        # Convert ke TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_pruned_model = converter.convert()
        
        tflite_pruned_path = os.path.join(config.MODELS_DIR, 'model_pruned.tflite')
        with open(tflite_pruned_path, 'wb') as f:
            f.write(tflite_pruned_model)
        
        print(f"✅ Pruned model saved: {pruned_path}")
        print(f"✅ Pruned TFLite model saved: {tflite_pruned_path}")
        
        # Compare sizes
        original_size = os.path.getsize(model_path)
        pruned_size = os.path.getsize(pruned_path)
        
        print(f"\nSize reduction: {(1 - pruned_size/original_size)*100:.1f}%")
        
        return final_model
        
    except ImportError:
        print("Install tensorflow_model_optimization first:")
        print("pip install tensorflow-model-optimization")
        return None

if __name__ == "__main__":
    prune_model()