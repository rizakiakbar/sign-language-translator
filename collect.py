import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

class ManualDataCollector:
    def __init__(self):
        # Setup MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Setup camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # width
        self.cap.set(4, 480)  # height
        
        # Data storage
        self.data_dir = "data/raw_coordinates"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Collection parameters
        self.current_letter = None
        self.samples_collected = 0
        self.total_samples = 0
        self.collection_active = False
        self.auto_collect = False
        self.collection_speed = 1  # samples per second
        
    def get_landmark_names(self):
        """Generate nama landmark untuk column CSV"""
        landmarks = []
        for i in range(21):  # 21 landmarks di MediaPipe Hands
            landmarks.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        landmarks.append('label')
        return landmarks
    
    def extract_landmarks(self, frame):
        """Extract koordinat landmark dari frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(landmarks)
                
                # Draw landmarks untuk visual feedback
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return landmarks_list, frame, hand_detected
    
    def save_sample(self, landmarks, letter):
        """Simpan sample ke CSV"""
        if len(landmarks) != 63:  # 21 landmarks * 3 coordinates
            return False
            
        # Buat DataFrame atau append ke existing
        csv_path = os.path.join(self.data_dir, f'{letter}_coordinates.csv')
        
        sample_data = landmarks + [letter]
        columns = self.get_landmark_names()
        
        df_new = pd.DataFrame([sample_data], columns=columns)
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
        
        return True
    
    def collect_for_letter(self, letter, num_samples=500):
        """Koleksi data untuk satu huruf"""
        self.current_letter = letter
        self.samples_collected = 0
        self.total_samples = num_samples
        
        print(f"\n🎯 Collecting data for letter: {letter}")
        print(f"Target: {num_samples} samples")
        print("Controls:")
        print("  SPACE - Collect one sample")
        print("  A     - Toggle auto collection (1 sample/second)") 
        print("  S     - Stop auto collection")
        print("  Q     - Finish and save")
        
        auto_start_time = 0
        collected_data = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror effect
            landmarks_list, processed_frame, hand_detected = self.extract_landmarks(frame)
            
            # Display information
            self.display_info(processed_frame, letter, hand_detected)
            
            # Auto collection logic
            if self.auto_collect and hand_detected:
                current_time = time.time()
                if current_time - auto_start_time >= 1.0 / self.collection_speed:
                    if landmarks_list and len(landmarks_list[0]) == 63:
                        if self.save_sample(landmarks_list[0], letter):
                            self.samples_collected += 1
                            collected_data.append(landmarks_list[0])
                            print(f"Auto-collected sample {self.samples_collected}/{num_samples}")
                        auto_start_time = current_time
            
            cv2.imshow('Manual Data Collection', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - manual collection
                if hand_detected and landmarks_list and len(landmarks_list[0]) == 63:
                    if self.save_sample(landmarks_list[0], letter):
                        self.samples_collected += 1
                        collected_data.append(landmarks_list[0])
                        print(f"Manual collected sample {self.samples_collected}/{num_samples}")
                        
            elif key == ord('a'):  # A - start auto collection
                if not self.auto_collect and hand_detected:
                    self.auto_collect = True
                    auto_start_time = time.time()
                    print("Auto collection STARTED")
                    
            elif key == ord('s'):  # S - stop auto collection
                if self.auto_collect:
                    self.auto_collect = False
                    print("Auto collection STOPPED")
                    
            elif key == ord('q'):  # Q - quit
                print(f"Finished collecting for {letter}. Total samples: {self.samples_collected}")
                break
                
            # Check if target reached
            if self.samples_collected >= num_samples:
                print(f"Target reached! Collected {self.samples_collected} samples for {letter}")
                break
        
        return collected_data
    
    def display_info(self, frame, letter, hand_detected):
        """Display information pada frame"""
        # Status hand detection
        hand_status = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        
        cv2.putText(frame, f"Letter: {letter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {self.samples_collected}/{self.total_samples}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, hand_status, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # Collection mode
        mode = "AUTO" if self.auto_collect else "MANUAL"
        mode_color = (0, 255, 255) if self.auto_collect else (255, 255, 0)
        cv2.putText(frame, f"Mode: {mode}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Controls reminder
        controls = [
            "CONTROLS:",
            "SPACE - Collect sample",
            "A - Auto collect", 
            "S - Stop auto",
            "Q - Finish letter"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (400, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def create_combined_dataset(self):
        """Gabungkan semua file CSV menjadi dataset lengkap"""
        print("\n📊 Combining all datasets...")
        
        all_dataframes = []
        letters = [chr(i) for i in range(65, 91)]  # A-Z
        
        for letter in letters:
            csv_path = os.path.join(self.data_dir, f'{letter}_coordinates.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_dataframes.append(df)
                print(f"  {letter}: {len(df)} samples")
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Buat direktori processed
            processed_dir = "data/processed"
            os.makedirs(processed_dir, exist_ok=True)
            
            combined_path = os.path.join(processed_dir, "complete_dataset.csv")
            combined_df.to_csv(combined_path, index=False)
            
            print(f"\n✅ Combined dataset saved: {combined_path}")
            print(f"📈 Total samples: {len(combined_df)}")
            print(f"🎯 Classes: {combined_df['label'].nunique()}")
            
            # Show class distribution
            print("\n📋 Class Distribution:")
            class_dist = combined_df['label'].value_counts().sort_index()
            for letter, count in class_dist.items():
                print(f"  {letter}: {count} samples")
                
            return combined_df
        else:
            print("❌ No data found to combine!")
            return None
    
    def run_collection_session(self):
        """Jalankan session koleksi data untuk semua huruf"""
        print("=== SIGN LANGUAGE COORDINATE DATA COLLECTION ===")
        print("Anda akan mengumpulkan data koordinat untuk setiap huruf A-Z")
        
        samples_per_letter = int(input("Masukkan jumlah sample per huruf (default 200): ") or 200)
        
        letters = [chr(i) for i in range(65, 91)]  # A-Z
        
        for letter in letters:
            print(f"\n{'='*50}")
            start_letter = input(f"Mulai koleksi untuk huruf {letter}? (y/n): ").lower()
            
            if start_letter == 'y':
                self.collect_for_letter(letter, samples_per_letter)
            else:
                print(f"Skipped letter {letter}")
                
            continue_session = input("\nLanjut ke huruf berikutnya? (y/n): ").lower()
            if continue_session != 'y':
                print("Collection session ended.")
                break
        
        # Combine datasets
        self.create_combined_dataset()
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    collector = ManualDataCollector()
    collector.run_collection_session()

if __name__ == "__main__":
    main()