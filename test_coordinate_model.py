import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import json
from src.data_loader import CoordinateDataLoader
from src.config import Config

def load_best_model():
    """Load model terbaik"""
    model_paths = [
        os.path.join(Config().MODELS_DIR, 'coordinate_model.h5'),  # Best model dari checkpoint
        os.path.join(Config().MODELS_DIR, 'coordinate_model_final.h5'),  # Final model
        os.path.join(Config().MODELS_DIR, 'best_model.h5')  # Fallback
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            return tf.keras.models.load_model(model_path)
    
    raise FileNotFoundError("No trained model found!")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    
    plt.title('Confusion Matrix - Coordinate-Based Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm

def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """Plot per-class metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    # Precision
    axes[0, 0].bar(class_names, precision, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Precision', fontweight='bold')
    axes[0, 0].set_xticklabels(class_names, rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Recall
    axes[0, 1].bar(class_names, recall, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Recall', fontweight='bold')
    axes[0, 1].set_xticklabels(class_names, rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # F1-Score
    axes[1, 0].bar(class_names, f1, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('F1-Score', fontweight='bold')
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Support
    axes[1, 1].bar(class_names, support, color='gold', alpha=0.7)
    axes[1, 1].set_title('Support (Number of Samples)', fontweight='bold')
    axes[1, 1].set_xticklabels(class_names, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to: {save_path}")
    
    plt.show()
    
    return metrics_df

def test_single_prediction(model, data_loader, sample_index=None):
    """Test prediction untuk single sample"""
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_loader.load_dataset()
    
    if sample_index is None:
        sample_index = np.random.randint(0, len(X_test))
    
    # Get sample
    sample = X_test[sample_index]
    true_label = y_test[sample_index]
    true_class = class_names[true_label]
    
    # Predict
    prediction = model.predict(sample.reshape(1, -1), verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(prediction[0])
    
    # Top 3 predictions
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_predictions = [(class_names[i], prediction[0][i]) for i in top_3_indices]
    
    print(f"\n=== SINGLE SAMPLE PREDICTION TEST ===")
    print(f"Sample Index: {sample_index}")
    print(f"True Label: {true_class} ({true_label})")
    print(f"Predicted: {predicted_class} (confidence: {confidence:.4f})")
    print(f"Correct: {true_class == predicted_class}")
    
    print(f"\nTop 3 Predictions:")
    for i, (cls, conf) in enumerate(top_3_predictions):
        print(f"  {i+1}. {cls}: {conf:.4f}")
    
    return true_class == predicted_class

def main():
    print("=== COORDINATE-BASED SIGN LANGUAGE MODEL TESTING ===")
    
    # Buat direktori results
    os.makedirs(Config().RESULTS_DIR, exist_ok=True)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python train_coordinate_model.py' first to train a model.")
        return
    
    # Load test data
    print("\n2. Loading test data...")
    data_loader = CoordinateDataLoader()
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_loader.load_dataset()
    except FileNotFoundError:
        print("Error: Dataset not found!")
        return
    
    # Evaluate model
    print("\n3. Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    print("\n4. Generating predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification Report
    print("\n5. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("\n6. Generating confusion matrix...")
    cm_path = os.path.join(Config().RESULTS_DIR, 'coordinate_confusion_matrix.png')
    cm = plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    
    # Per-class Metrics
    print("\n7. Calculating per-class metrics...")
    metrics_path = os.path.join(Config().RESULTS_DIR, 'coordinate_per_class_metrics.png')
    metrics_df = plot_per_class_metrics(y_test, y_pred, class_names, metrics_path)
    
    # Test single predictions
    print("\n8. Testing single predictions...")
    correct_predictions = 0
    for i in range(5):  # Test 5 random samples
        if test_single_prediction(model, data_loader):
            correct_predictions += 1
    
    print(f"\nSingle prediction test: {correct_predictions}/5 correct")
    
    # Save detailed test results
    print("\n9. Saving test results...")
    test_results = {
        'overall_metrics': {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        },
        'per_class_metrics': metrics_df.to_dict('records'),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    }
    
    results_path = os.path.join(Config().RESULTS_DIR, 'coordinate_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Save text summary
    summary_path = os.path.join(Config().RESULTS_DIR, 'coordinate_test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== COORDINATE-BASED MODEL TEST RESULTS ===\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"\nTest results saved to: {results_path}")
    print(f"Test summary saved to: {summary_path}")
    
    print("\n=== TESTING COMPLETED ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()