import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from src.data_loader import CoordinateDataLoader
from src.model_training import CoordinateModel
from src.config import Config

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")
    
    plt.show()

def save_training_report(history, test_accuracy, test_loss, class_names, save_dir):
    """Save training report"""
    report = {
        'training_history': {
            'final_training_accuracy': history.history['accuracy'][-1],
            'final_validation_accuracy': history.history['val_accuracy'][-1],
            'final_training_loss': history.history['loss'][-1],
            'final_validation_loss': history.history['val_loss'][-1],
            'best_validation_accuracy': max(history.history['val_accuracy']),
            'best_validation_loss': min(history.history['val_loss'])
        },
        'test_results': {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        },
        'model_info': {
            'num_classes': len(class_names),
            'input_features': Config().NUM_FEATURES,
            'classes': class_names.tolist()
        },
        'training_parameters': {
            'epochs': Config().EPOCHS,
            'batch_size': Config().BATCH_SIZE,
            'learning_rate': Config().LEARNING_RATE
        }
    }
    
    # Save as JSON
    report_path = os.path.join(save_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save as text
    text_report_path = os.path.join(save_dir, 'training_summary.txt')
    with open(text_report_path, 'w') as f:
        f.write("=== COORDINATE-BASED SIGN LANGUAGE MODEL TRAINING REPORT ===\n\n")
        f.write("MODEL INFORMATION:\n")
        f.write(f"- Input Features: {report['model_info']['input_features']}\n")
        f.write(f"- Number of Classes: {report['model_info']['num_classes']}\n")
        f.write(f"- Classes: {', '.join(report['model_info']['classes'])}\n\n")
        
        f.write("TRAINING PARAMETERS:\n")
        f.write(f"- Epochs: {report['training_parameters']['epochs']}\n")
        f.write(f"- Batch Size: {report['training_parameters']['batch_size']}\n")
        f.write(f"- Learning Rate: {report['training_parameters']['learning_rate']}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"- Final Training Accuracy: {report['training_history']['final_training_accuracy']:.4f}\n")
        f.write(f"- Final Validation Accuracy: {report['training_history']['final_validation_accuracy']:.4f}\n")
        f.write(f"- Best Validation Accuracy: {report['training_history']['best_validation_accuracy']:.4f}\n")
        f.write(f"- Final Training Loss: {report['training_history']['final_training_loss']:.4f}\n")
        f.write(f"- Final Validation Loss: {report['training_history']['final_validation_loss']:.4f}\n")
        f.write(f"- Best Validation Loss: {report['training_history']['best_validation_loss']:.4f}\n\n")
        
        f.write("TEST RESULTS:\n")
        f.write(f"- Test Accuracy: {report['test_results']['test_accuracy']:.4f}\n")
        f.write(f"- Test Loss: {report['test_results']['test_loss']:.4f}\n")
    
    print(f"Training report saved to: {report_path}")
    print(f"Training summary saved to: {text_report_path}")

def main():
    print("=== COORDINATE-BASED SIGN LANGUAGE MODEL TRAINING ===")
    
    # Initialize config
    config = Config()
    
    # Buat direktori
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_loader = CoordinateDataLoader()
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_loader.load_dataset()
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Classes: {class_names}")
        
    except FileNotFoundError:
        print("Error: Dataset not found!")
        print("Please run 'python collect_coordinates.py' first to collect data.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Build model
    print("\n2. Building model...")
    model = CoordinateModel()
    model.build_model()
    model.model.summary()
    
    # Train model - ✅ PASS data_loader ke training function
    print("\n3. Training model...")
    history = model.train(X_train, y_train, X_val, y_val, data_loader)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Plot training history
    print("\n5. Generating plots...")
    plot_path = os.path.join(config.RESULTS_DIR, 'coordinate_training_history.png')
    plot_training_history(history, plot_path)
    
    # Save training report
    print("\n6. Saving training report...")
    save_training_report(history, test_accuracy, test_loss, class_names, config.RESULTS_DIR)
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'coordinate_model_final.h5')
    model.model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Model ready for real-time detection!")

if __name__ == "__main__":
    main()