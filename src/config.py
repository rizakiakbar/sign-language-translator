import os

class Config:
    # Path direktori
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_COORD_DIR = os.path.join(DATA_DIR, 'raw_coordinates')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')  # ✅ TAMBAH INI
    
    # Parameter MediaPipe Hands
    MAX_HANDS = 1
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Parameter dataset
    NUM_LANDMARKS = 21  # 21 landmark tangan MediaPipe
    NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z untuk setiap landmark
    NUM_CLASSES = 26  # A-Z
    
    # Parameter model
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Label huruf A-Z
    LETTERS = [chr(i) for i in range(65, 91)]  # A-Z
    
    # File paths
    COORDINATE_CSV = os.path.join(PROCESSED_DIR, 'complete_dataset.csv')
    MODEL_PATH = os.path.join(MODELS_DIR, 'coordinate_model.h5')