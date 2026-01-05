import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from src.config import Config

class CoordinateDataLoader:
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, test_size=0.2, val_size=0.2):
        """Load dan preprocess dataset koordinat"""
        print(f"Looking for dataset at: {self.config.COORDINATE_CSV}")
        print(f"File exists: {os.path.exists(self.config.COORDINATE_CSV)}")
        
        if not os.path.exists(self.config.COORDINATE_CSV):
            # Cek apa ada file di direktori
            processed_dir = self.config.PROCESSED_DIR
            print(f"Contents of {processed_dir}:")
            if os.path.exists(processed_dir):
                for file in os.listdir(processed_dir):
                    print(f"  - {file}")
            raise FileNotFoundError(f"Dataset not found: {self.config.COORDINATE_CSV}")
        
        # Load data
        df = pd.read_csv(self.config.COORDINATE_CSV)
        print(f"Dataset loaded: {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Check for missing values
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Pisahkan features dan labels
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns].values
        y = df['label'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels: {np.unique(y)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Encoded labels: {np.unique(y_encoded)}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split data: train -> 60%, val -> 20%, test -> 20%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return (X_train_scaled, y_train, X_val_scaled, y_val, 
                X_test_scaled, y_test, self.label_encoder.classes_)
    
    def preprocess_single_sample(self, landmarks):
        """Preprocess single sample untuk prediction"""
        if len(landmarks) != self.config.NUM_FEATURES:
            raise ValueError(f"Expected {self.config.NUM_FEATURES} features, got {len(landmarks)}")
        
        landmarks_array = np.array(landmarks).reshape(1, -1)
        return self.scaler.transform(landmarks_array)