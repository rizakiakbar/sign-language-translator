# **TUTORIAL MENJALANKAN PROJECT PENERJEMAH BAHASA ISYARAT**

## **📦 INSTALASI DAN SETUP AWAL**

### **1. KLONING REPOSITORY**

```bash
# Clone repository dari GitHub
git clone https://https://github.com/rizakiakbar/sign-language-translator.git
cd sign-language-translator
```

### **2. BUAT VIRTUAL ENVIRONMENT (REKOMENDASI)**

```bash
# Untuk Windows
python -m venv sign_env
sign_env\Scripts\activate

# Untuk Mac/Linux
python3 -m venv sign_env
source sign_env/bin/activate
```

### **3. INSTAL DEPENDENSI**

```bash
# Install semua requirements
pip install -r requirements.txt

# Atau install manual jika requirements.txt tidak ada
pip install joblib==1.5.2
pip install matplotlib==3.8.4
pip install mediapipe==0.10.9
pip install numpy==2.4.0
pip install opencv_contrib_python==4.11.0.86
pip install opencv_python==4.9.0.80
pip install pandas==2.3.3
pip install psutil==7.1.3
pip install PyQt5==5.15.11
pip install pyqt5_sip==12.17.2
pip install scikit_learn==1.8.0
pip install seaborn==0.13.2
pip install tensorflow==2.15.0
pip install tensorflow_intel==2.15.0
pip install tensorflow_model_optimization==0.8.0
```

## **🚀 CARA MENJALANKAN APLIKASI**

### **OPTION 1: JIKA SUDAH ADA MODEL TERLATIH**

```bash
# 1. Pastikan model sudah ada di folder data/models/
#    - coordinate_model.h5
#    - scaler.pkl

# 2. Jalankan aplikasi GUI
python app/multimedia_app.py

```

### **OPTION 2: JIKA BELUM ADA MODEL (BUAT DARI AWAL)**

```bash
# Langkah 1: Koleksi data training
python collect_coordinate.py

# Langkah 2: Training model
python train_coordinate_model.py

# Langkah 3: Testing model
python test_coordinate_model.py

# Langkah 4: Jalankan aplikasi
python realtime_detection_coordinate.py
```

## **📊 STRUKTUR PROJECT YANG HARUS ADA**

```
sign_language_translator/
├── 📁 data/
│   ├── 📁 models/                 # WAJIB: Tempat model
│   │   ├── coordinate_model.h5    # Model neural network
│   │   └── scaler.pkl             # Scaler untuk preprocessing
│   ├── 📁 processed/              # Dataset yang diproses
│   └── 📁 raw_coordinates/        # Data mentah
├── 📁 src/                        # Source code
├── 📁 app/                        # GUI application
├── 📁 results/                    # Hasil training
└── 📄 requirements.txt            # Dependencies
```

## **🔧 TROUBLESHOOTING UMUM**

### **ERROR: "No module named 'src'"**

```bash
# Tambahkan current directory ke Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Atau jalankan dari root directory project
cd /path/to/sign_language_translator
```

### **ERROR: Camera not detected**

```python
# Coba ganti camera index
cap = cv2.VideoCapture(0)  # Ganti 0 dengan 1, 2, dst

# Check camera list
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

### **ERROR: TensorFlow/CUDA issues**

```bash
# Jika pakai GPU, pastikan CUDA terinstall
# Atau force CPU-only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### **ERROR: PyQt5 not found**

```bash
# Install PyQt5 khusus
pip install PyQt5==5.15.9

# Atau install dari PyPI alternatif
pip install PyQt5-sip PyQt5-Qt5 PyQt5-stubs
```

## **🎯 MODE PENGGUNAAN APLIKASI**

### **Mode 1: Real-time Detection**

```
1. Buka aplikasi
2. Klik "▶ Start Detection"
3. Tunjukkan tangan dengan isyarat huruf A-Z di depan kamera
4. Sistem akan menampilkan prediksi dan confidence level
5. Tekan 'q' untuk keluar
```

### **Mode 2: Learning Mode**

```
1. Pilih tab "Learning Mode"
2. Klik huruf yang ingin dipelajari (A-Z)
3. Lihat tutorial gambar dan petunjuk
4. Praktikkan di depan kamera
5. Cek akurasi di tab "Detection"
```

### **Mode 3: Analytics**

```
1. Pilih tab "Analytics"
2. Lihat grafik performa sistem
3. Monitor accuracy trends
4. Analisis distribusi huruf yang terdeteksi
```

## **📝 PANDUAN CEPAT UNTUK PRESENTASI**

### **Setup Cepat (5 Menit)**

```bash
# 1. Clone repo
git clone [repository-url]

# 2. Install dependencies (yang essential saja)
pip install opencv-python mediapipe tensorflow numpy

# 3. Download pre-trained model
#    Letakkan di: data/models/coordinate_model.h5
#    Letakkan di: data/models/scaler.pkl

# 4. Jalankan
python realtime_detection_coordinate.py
```

### **Demo Script untuk Presentasi**

```bash
# Demo 1: Show hand detection
python -c "import cv2; print('Camera test: OK')"

# Demo 2: Full application
python app/multimedia_app.py
```

## **💾 BACKUP DAN RESTORE**

### **Backup Model**

```bash
# Backup semua model dan data
mkdir backup_$(date +%Y%m%d)
cp -r data/models backup_$(date +%Y%m%d)/
cp -r data/processed backup_$(date +%Y%m%d)/
tar -czf backup_$(date +%Y%m%d).tar.gz backup_$(date +%Y%m%d)/
```

### **Restore dari Backup**

```bash
# Extract backup
tar -xzf backup_YYYYMMDD.tar.gz

# Restore ke project
cp -r backup_YYYYMMDD/models/* data/models/
cp -r backup_YYYYMMDD/processed/* data/processed/
```

## **🔍 DEBUGGING TIPS**

### **Enable Debug Mode**

```python
# Tambahkan di awal script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Check System Info**

```bash
# Check Python version
python --version

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"

# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera ERROR')"
```

### **Common Issues & Fixes**

```
ISSUE: Aplikasi crash saat start
FIX: Pastikan semua dependencies terinstall

ISSUE: Prediction tidak akurat
FIX: Pastikan lighting cukup, tangan jelas terlihat

ISSUE: FPS rendah
FIX: Turunkan resolusi kamera di code (640x480)

ISSUE: Model tidak load
FIX: Check path model, pastikan file .h5 ada
```

## **📱 MINIMAL REQUIREMENTS**

### **Hardware Minimum**

```
• CPU: Intel i5 generasi 8 atau setara
• RAM: 8GB
• Storage: 10GB free space
• Camera: 720p webcam
• OS: Windows 10, macOS 10.15+, Ubuntu 18.04+
```

### **Software Requirements**

```
• Python 3.8 atau lebih baru
• pip package manager
• Webcam driver terinstall
• Visual Studio Code (rekomendasi untuk editing)
```

## **🎮 SHORTCUT KEYBOARD**

```
q          - Keluar dari aplikasi
c          - Capture screenshot
r          - Reset prediction history
Space      - Start/Stop detection
Tab        - Switch between modes
1,2,3      - Switch tabs (1=Detection, 2=Learning, 3=Analytics)
ESC        - Close current dialog
```

## **📈 PERFORMANCE OPTIMIZATION**

### **Untuk FPS lebih tinggi:**

```python
# Di code, ubah:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Dari 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Dari 480
```

### **Untuk akurasi lebih baik:**

```
1. Gunakan lighting frontal (tidak backlight)
2. Background polos (tidak patterned)
3. Jarak tangan 30-50cm dari kamera
4. Posisi tangan sejajar dengan bahu
5. Hindari gerakan terlalu cepat
```

## **📚 RESOURCES TAMBAHAN**

### **Links Penting:**

- Repository GitHub: https://github.com/rizakiakbar/sign-language-translator
- Dokumentasi MediaPipe: https://google.github.io/mediapipe/
- Dokumentasi TensorFlow: https://www.tensorflow.org/
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/

### **Dataset Alternatif:**

- ASL Alphabet Dataset: https://www.kaggle.com/grassknoted/asl-alphabet
- Custom dataset bisa dibuat dengan collect_coordinates.py

## **🆘 SUPPORT & CONTACT**

### **Jika mengalami masalah:**

1. Cek file `error_log.txt` di folder project
2. Ambil screenshot error message
3. Cek apakah semua dependencies terinstall
4. Pastikan model files ada di folder yang benar

### **Contact Developer:**

- Email: [developer-email]
- GitHub Issues: [repo-issues-link]
- Documentation: [docs-link]

---

## **🚨 EMERGENCY FIXES**

### **Jika aplikasi tidak bisa jalan sama sekali:**

```bash
# Clean install
pip uninstall -y tensorflow opencv-python mediapipe PyQt5
pip install --upgrade pip
pip install tensorflow opencv-python mediapipe PyQt5

# Run minimal version
python scripts/minimal_demo.py
```

### **Minimal Demo Script (minimal_demo.py):**

```python
import cv2
print("OpenCV test - Show camera feed")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

**🎉 SELAMAT MENCOBA!**  
Jika ada masalah, cek troubleshooting section atau buat issue di repository.
