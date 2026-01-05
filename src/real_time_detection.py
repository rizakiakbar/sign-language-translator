import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from .config import Config

class RealTimeCoordinateDetector:
    def __init__(self, model_path):
        self.config = Config()
        self.model = tf.keras.models.load_model(model_path)
        
        # Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Setup camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # Prediction smoothing
        self.prediction_history = []
        self.history_size = 5
    
    def extract_and_predict(self, frame):
        """Extract landmarks dan lakukan prediction"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        prediction = "None"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Preprocess dan predict
                if len(landmarks) == self.config.NUM_FEATURES:
                    from data_loader import CoordinateDataLoader
                    data_loader = CoordinateDataLoader()
                    
                    try:
                        processed_landmarks = data_loader.preprocess_single_sample(landmarks)
                        predictions = self.model.predict(processed_landmarks, verbose=0)
                        
                        predicted_class = np.argmax(predictions[0])
                        current_confidence = np.max(predictions[0])
                        current_prediction = self.config.LETTERS[predicted_class]
                        
                        # Smooth prediction
                        prediction, confidence = self.smooth_prediction(
                            current_prediction, current_confidence
                        )
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame, prediction, confidence
    
    def smooth_prediction(self, current_pred, current_conf):
        """Smooth prediction menggunakan history"""
        self.prediction_history.append((current_pred, current_conf))
        
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        if len(self.prediction_history) == self.history_size:
            # Ambil prediction dengan confidence tertinggi
            best_pred = max(self.prediction_history, key=lambda x: x[1])
            return best_pred[0], best_pred[1]
        
        return current_pred, current_conf
    
    def run_detection(self):
        """Jalankan real-time detection"""
        print("Starting coordinate-based detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            processed_frame, prediction, confidence = self.extract_and_predict(frame)
            
            # Display results
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            
            cv2.putText(processed_frame, f"Pred: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(processed_frame, f"Conf: {confidence:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(processed_frame, "Press 'q' to quit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Coordinate-Based Sign Language Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()