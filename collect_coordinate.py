from src.coordinate_extractor import CoordinateExtractor
from src.config import Config
import os

def main():
    # Buat direktori
    os.makedirs(Config().RAW_COORD_DIR, exist_ok=True)
    
    extractor = CoordinateExtractor()
    
    print("=== SIGN LANGUAGE COORDINATE COLLECTION ===")
    print("Collecting coordinates for letters A-Z")
    
    # Koleksi data untuk setiap huruf
    for letter in Config().LETTERS:
        print(f"\n=== COLLECTING FOR LETTER: {letter} ===")
        extractor.collect_coordinates_for_letter(letter, num_samples=200)
    
    # Gabungkan semua dataset
    print("\n=== COMBINING DATASETS ===")
    extractor.create_combined_dataset()

if __name__ == "__main__":
    main()