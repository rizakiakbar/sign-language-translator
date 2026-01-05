import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import joblib
import sys
from src.config import Config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class RealTimeCoordinateDetector:
    def __init__(self, model_path=None):
        print("Initializing RealTimeCoordinateDetector...")
        self.config = Config()
        
        # Load model
        if model_path is None:
            model_path = os.path.join(self.config.MODELS_DIR, 'coordinate_model.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.config.MODELS_DIR, 'coordinate_model_final.h5')
        
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file not found at {model_path}")
            print("Please run 'python train_coordinate_model.py' first to train the model.")
            sys.exit(1)
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            sys.exit(1)
        
        # Load scaler
        try:
            self.scaler = self.load_scaler()
            if self.scaler is None:
                print("❌ ERROR: Scaler not found! Please train the model first.")
                sys.exit(1)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading scaler: {e}")
            sys.exit(1)
        
        # Setup MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.MAX_HANDS,
                min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("✅ MediaPipe Hands initialized")
        except Exception as e:
            print(f"❌ ERROR initializing MediaPipe: {e}")
            sys.exit(1)
        
        # Setup camera
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ ERROR: Cannot open camera")
                sys.exit(1)
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✅ Camera initialized successfully")
        except Exception as e:
            print(f"❌ ERROR initializing camera: {e}")
            sys.exit(1)
        
        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 7
        self.current_prediction = "None"
        self.current_confidence = 0.0
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ RealTimeCoordinateDetector initialized successfully!")
    
    def load_scaler(self):
        """Load scaler yang sudah di-fit dari file"""
        scaler_path = os.path.join(self.config.MODELS_DIR, 'scaler.pkl')
        print(f"Looking for scaler at: {scaler_path}")
        
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded from file")
                return scaler
            except Exception as e:
                print(f"Error loading scaler: {e}")
        
        # Jika scaler tidak ada, coba buat dari data loader
        try:
            from src.data_loader import CoordinateDataLoader
            data_loader = CoordinateDataLoader()
            # Load dataset untuk mem-fit scaler
            X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_loader.load_dataset()
            # Simpan scaler untuk future use
            os.makedirs(self.config.MODELS_DIR, exist_ok=True)
            joblib.dump(data_loader.scaler, scaler_path)
            print("✅ Scaler created and saved from dataset")
            return data_loader.scaler
        except Exception as e:
            print(f"Error creating scaler from dataset: {e}")
            return None
    
    def extract_landmarks(self, image):
        """Extract hand landmarks dari frame"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            landmarks = []
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract semua landmarks (x, y, z)
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Draw landmarks pada frame
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            return landmarks, image, hand_detected
        except Exception as e:
            print(f"Error in extract_landmarks: {e}")
            return [], image, False
    
    def preprocess_landmarks(self, landmarks):
        """Preprocess landmarks untuk prediction"""
        if len(landmarks) != self.config.NUM_FEATURES:
            return None
        
        try:
            # Convert to numpy array dan reshape
            landmarks_array = np.array(landmarks).reshape(1, -1)
            
            # Transform menggunakan scaler yang sudah di-fit
            processed_landmarks = self.scaler.transform(landmarks_array)
            
            return processed_landmarks
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_from_landmarks(self, landmarks):
        """Predict huruf dari landmarks"""
        if len(landmarks) != self.config.NUM_FEATURES:
            return "None", 0.0
        
        try:
            # Preprocess landmarks
            processed_landmarks = self.preprocess_landmarks(landmarks)
            
            if processed_landmarks is None:
                return "None", 0.0
            
            # Predict
            predictions = self.model.predict(processed_landmarks, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            predicted_letter = self.config.LETTERS[predicted_class_idx]
            
            return predicted_letter, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "None", 0.0
    
    def smooth_prediction(self, new_prediction, new_confidence):
        """Smooth prediction menggunakan moving average"""
        self.prediction_history.append((new_prediction, new_confidence))
        
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Jika ada cukup history, gunakan voting dengan confidence
        if len(self.prediction_history) >= 3:
            # Cari prediction dengan confidence tertinggi
            best_pred = max(self.prediction_history, key=lambda x: x[1])
            self.current_prediction = best_pred[0]
            self.current_confidence = best_pred[1]
        else:
            self.current_prediction = new_prediction
            self.current_confidence = new_confidence
        
        return self.current_prediction, self.current_confidence
    
    def draw_interface(self, frame, prediction, confidence, hand_detected):
        """Draw interface pada frame"""
        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            end_time = time.time()
            self.fps = 30 / (end_time - self.start_time)
            self.start_time = end_time
        
        # Status hand detection
        detection_status = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
        detection_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        
        # Prediction color based on confidence
        if confidence > 0.8:
            pred_color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.6:
            pred_color = (0, 255, 255)  # Yellow - medium confidence
        else:
            pred_color = (0, 165, 255)  # Orange - low confidence
        
        # Draw detection status
        cv2.putText(frame, f"Status: {detection_status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
        
        # Draw prediction
        cv2.putText(frame, f"Prediction: {prediction}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.3f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 180
        fill_width = int(confidence * bar_width)
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        # Confidence level
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), 
                     pred_color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
        
        cv2.putText(frame, "Confidence Level", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Instructions:",
            "1. Show one hand in the frame",
            "2. Make sign language gestures",
            "3. Press 'q' to quit",
            "4. Press 'c' to capture frame"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (10, 230 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self):
        """Jalankan real-time detection"""
        print("=== COORDINATE-BASED SIGN LANGUAGE DETECTION ===")
        print("Starting real-time detection...")
        print("Press 'q' to quit")
        print("Press 'c' to capture current frame")
        print("Press 'r' to reset prediction history")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks, processed_frame, hand_detected = self.extract_landmarks(frame)
                
                prediction = "None"
                confidence = 0.0
                
                if hand_detected and len(landmarks) == self.config.NUM_FEATURES:
                    # Predict dari landmarks
                    new_prediction, new_confidence = self.predict_from_landmarks(landmarks)
                    
                    # Smooth prediction
                    prediction, confidence = self.smooth_prediction(new_prediction, new_confidence)
                
                # Draw interface
                self.draw_interface(processed_frame, prediction, confidence, hand_detected)
                
                # Display frame
                cv2.imshow('Coordinate-Based Sign Language Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Capture frame
                    timestamp = int(time.time())
                    capture_path = os.path.join(self.config.RESULTS_DIR, f'capture_{timestamp}.jpg')
                    cv2.imwrite(capture_path, processed_frame)
                    print(f"Frame captured: {capture_path}")
                elif key == ord('r'):
                    # Reset prediction history
                    self.prediction_history = []
                    print("Prediction history reset")
        
        except Exception as e:
            print(f"Error during detection: {e}")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped")

def main():
    try:
        detector = RealTimeCoordinateDetector()
        detector.run_detection()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check:")
        print("1. Camera is connected and working")
        print("2. Model files exist in data/models/")
        print("3. All dependencies are installed")

if __name__ == "__main__":
    main()