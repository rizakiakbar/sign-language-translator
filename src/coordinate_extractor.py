import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from .config import Config

class CoordinateExtractor:
    def __init__(self):
        self.config = Config()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def extract_landmarks(self, image):
        """Extract landmarks dari image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(landmarks)
                
                # Draw landmarks (optional untuk visualization)
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return landmarks_list, image
    
    def collect_coordinates_for_letter(self, letter, num_samples=1000):
        """Koleksi koordinat untuk huruf tertentu"""
        cap = cv2.VideoCapture(0)
        collected_data = []
        count = 0
        
        print(f"Koleksi data untuk huruf: {letter}")
        print("Tekan 's' untuk simpan, 'q' untuk keluar")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks_list, processed_frame = self.extract_landmarks(frame)
            
            # Jika tangan terdeteksi
            if landmarks_list and len(landmarks_list[0]) == 63:  # 21 landmarks * 3
                # Tampilkan preview
                cv2.putText(processed_frame, f"Letter: {letter}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Samples: {count}/{num_samples}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Press 's' to save", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Coordinate Collection', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Simpan koordinat + label
                    sample_data = landmarks_list[0] + [letter]
                    collected_data.append(sample_data)
                    count += 1
                    print(f"Saved sample {count}/{num_samples}")
                    
                elif key == ord('q'):
                    break
            else:
                # Tampilkan instruksi jika tangan tidak terdeteksi
                cv2.putText(frame, "Hand not detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Show your hand in the frame", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Coordinate Collection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Simpan ke CSV
        if collected_data:
            self.save_to_csv(collected_data, letter)
            
        return collected_data
    
    def save_to_csv(self, data, letter):
        """Simpan data ke CSV"""
        # Buat column names
        columns = []
        for i in range(21):  # 21 landmarks
            columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        columns.append('label')
        
        # Buat DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Simpan file
        os.makedirs(self.config.RAW_COORD_DIR, exist_ok=True)
        csv_path = os.path.join(self.config.RAW_COORD_DIR, f'{letter}_coordinates.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")
        
        return df
    
    def create_combined_dataset(self):
        """Gabungkan semua file CSV menjadi dataset lengkap"""
        all_data = []
        
        for letter in self.config.LETTERS:
            csv_path = os.path.join(self.config.RAW_COORD_DIR, f'{letter}_coordinates.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_data.append(df)
                print(f"Loaded {len(df)} samples for {letter}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            os.makedirs(self.config.PROCESSED_DIR, exist_ok=True)
            combined_df.to_csv(self.config.COORDINATE_CSV, index=False)
            print(f"Combined dataset saved to: {self.config.COORDINATE_CSV}")
            print(f"Total samples: {len(combined_df)}")
            return combined_df
        else:
            print("No data found!")
            return None