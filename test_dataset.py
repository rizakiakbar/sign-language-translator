import os
import pandas as pd
from src.config import Config

def check_dataset():
    """Check dataset structure and contents"""
    config = Config()
    
    print("=== DATASET CHECK ===")
    print(f"PROCESSED_DIR: {config.PROCESSED_DIR}")
    print(f"COORDINATE_CSV: {config.COORDINATE_CSV}")
    
    # Check if directory exists
    print(f"\n1. Checking directories...")
    print(f"PROCESSED_DIR exists: {os.path.exists(config.PROCESSED_DIR)}")
    print(f"COORDINATE_CSV exists: {os.path.exists(config.COORDINATE_CSV)}")
    
    if os.path.exists(config.PROCESSED_DIR):
        print(f"Files in PROCESSED_DIR:")
        for file in os.listdir(config.PROCESSED_DIR):
            file_path = os.path.join(config.PROCESSED_DIR, file)
            print(f"  - {file} (size: {os.path.getsize(file_path)} bytes)")
    
    # Check CSV file
    if os.path.exists(config.COORDINATE_CSV):
        print(f"\n2. Checking CSV file...")
        try:
            df = pd.read_csv(config.COORDINATE_CSV)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 5 rows:")
            print(df.head())
            print(f"\nLabel distribution:")
            print(df['label'].value_counts().sort_index())
            
            # Check expected features
            expected_features = config.NUM_FEATURES
            actual_features = len([col for col in df.columns if col != 'label'])
            print(f"\nExpected features: {expected_features} (21 landmarks * 3 coordinates)")
            print(f"Actual features: {actual_features}")
            
            if actual_features != expected_features:
                print(f"❌ FEATURE COUNT MISMATCH!")
                print(f"Check your data collection process")
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"\n❌ CSV file not found at: {config.COORDINATE_CSV}")
        print(f"Please run: python collect_coordinates.py")

def check_raw_data():
    """Check raw coordinate files"""
    config = Config()
    
    print(f"\n3. Checking raw data...")
    print(f"RAW_COORD_DIR: {config.RAW_COORD_DIR}")
    print(f"RAW_COORD_DIR exists: {os.path.exists(config.RAW_COORD_DIR)}")
    
    if os.path.exists(config.RAW_COORD_DIR):
        csv_files = [f for f in os.listdir(config.RAW_COORD_DIR) if f.endswith('.csv')]
        print(f"Raw CSV files found: {len(csv_files)}")
        for file in csv_files:
            file_path = os.path.join(config.RAW_COORD_DIR, file)
            df = pd.read_csv(file_path)
            print(f"  - {file}: {len(df)} samples")

if __name__ == "__main__":
    check_dataset()
    check_raw_data()