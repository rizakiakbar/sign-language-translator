# optimize_model.py
import tensorflow as tf
from tensorflow import keras
import os
from src.config import Config

def optimize_model_for_production():
    """Optimize model untuk deployment"""
    config = Config()
    
    # Load model
    model_path = os.path.join(config.MODELS_DIR, 'coordinate_model.h5')
    model = keras.models.load_model(model_path)
    
    # 1. CONVERT KE TFLITE (untuk mobile/edge devices)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # FP16 quantization
    converter.experimental_new_converter = True
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(config.MODELS_DIR, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved: {tflite_path}")
    print(f"Original size: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"TFLite size: {len(tflite_model) / 1024:.2f} KB")
    
    # 2. SAVE DALAM FORMAT SAVED_MODEL
    saved_model_path = os.path.join(config.MODELS_DIR, 'saved_model')
    model.save(saved_model_path, save_format='tf')
    print(f"✅ SavedModel format: {saved_model_path}")
    
    return tflite_model

def benchmark_model():
    """Benchmark model performance"""
    import time
    import numpy as np
    
    config = Config()
    
    # Load model
    model_path = os.path.join(config.MODELS_DIR, 'coordinate_model.h5')
    model = keras.models.load_model(model_path)
    
    # Test inference speed
    dummy_input = np.random.randn(1, config.NUM_FEATURES).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    # Benchmark
    num_tests = 100
    start_time = time.time()
    
    for _ in range(num_tests):
        _ = model.predict(dummy_input, verbose=0)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_tests
    
    print(f"\n=== MODEL BENCHMARK ===")
    print(f"Total predictions: {num_tests}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS potential: {1/avg_time:.1f}")
    
    return avg_time

if __name__ == "__main__":
    optimize_model_for_production()
    benchmark_model()