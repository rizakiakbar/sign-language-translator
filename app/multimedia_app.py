# app/multimedia_gui.py
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import tensorflow as tf
import mediapipe as mp
import joblib
import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import psutil

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Updated style sheet for better fullscreen display on Windows
APP_STYLE = """
QMainWindow {
    background-color: #1a1a2e;
    border: none;
}

QFrame {
    background-color: #16213e;
    border-radius: 15px;
    border: 2px solid #0f3460;
}

QLabel {
    color: #e6e6e6;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QPushButton {
    background-color: #0f3460;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: bold;
    min-height: 40px;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #1a5f7a;
    border: 2px solid #57cc99;
}

QPushButton:pressed {
    background-color: #0a2647;
}

QPushButton#startBtn {
    background-color: #57cc99;
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
}

QPushButton#startBtn:hover {
    background-color: #80ed99;
}

QPushButton#captureBtn {
    background-color: #ff9a3c;
    color: #ffffff;
}

QPushButton#captureBtn:hover {
    background-color: #ffcc29;
}

QPushButton#tutorialBtn {
    background-color: #6a67ce;
    color: white;
}

QProgressBar {
    border: 2px solid #0f3460;
    border-radius: 10px;
    text-align: center;
    height: 25px;
    background-color: #1a1a2e;
    color: white;
}

QProgressBar::chunk {
    background-color: qlineargradient(
        spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, 
        stop:0 #57cc99, stop:1 #80ed99
    );
    border-radius: 8px;
}

QListWidget {
    background-color: #1a1a2e;
    border: 2px solid #0f3460;
    border-radius: 10px;
    color: #e6e6e6;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #0f3460;
}

QListWidget::item:selected {
    background-color: #0f3460;
    color: #57cc99;
}

QComboBox {
    background-color: #1a1a2e;
    color: white;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 5px;
    min-height: 30px;
}

QComboBox:hover {
    border-color: #57cc99;
}

QSlider::groove:horizontal {
    height: 10px;
    background: #0f3460;
    border-radius: 5px;
}

QSlider::handle:horizontal {
    background: #57cc99;
    width: 20px;
    height: 20px;
    margin: -5px 0;
    border-radius: 10px;
}

QTabWidget::pane {
    border: 2px solid #0f3460;
    border-radius: 10px;
    background-color: #16213e;
    padding: 10px;
}

QTabBar::tab {
    background-color: #0f3460;
    color: white;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    min-width: 120px;
}

QTabBar::tab:selected {
    background-color: #57cc99;
    color: #1a1a2e;
    font-weight: bold;
}

QTextEdit {
    background-color: #1a1a2e;
    color: #e6e6e6;
    border: 2px solid #0f3460;
    border-radius: 10px;
    padding: 10px;
    font-family: 'Segoe UI', sans-serif;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

QDialog {
    background-color: #1a1a2e;
    border: 2px solid #57cc99;
    border-radius: 15px;
}
"""

class TutorialDialog(QDialog):
    """Dialog untuk menampilkan tutorial dengan gambar"""
    def __init__(self, letter, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.parent_app = parent  # Simpan referensi ke aplikasi utama
        self.setWindowTitle(f"📚 Tutorial Huruf {letter}")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(APP_STYLE)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(f"TUTORIAL HURUF {self.letter}")
        
        # Gunakan font config dari parent jika ada
        font_size = 28
        if hasattr(self.parent_app, 'font_config'):
            font_size = self.parent_app.font_config.get('tutorial_header', 28)
        
        header.setStyleSheet(f"""
            font-size: {font_size}px;
            font-weight: bold;
            color: #57cc99;
            padding: 15px;
            border-bottom: 3px solid #57cc99;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Content area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Image placeholder
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border: 3px solid #57cc99;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        image_layout = QVBoxLayout()
        
        # Placeholder untuk gambar tutorial
        image_label = QLabel()
        image_label.setFixedSize(400, 300)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px dashed #57cc99;
                border-radius: 10px;
                color: #80ed99;
                font-size: 16px;
            }
        """)
        
        # Coba load gambar jika ada
        image_path = os.path.join(project_root, 'data', 'tutorial_images', f'{self.letter}.jpg')
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.setText("")
            else:
                image_label.setText(f"Gambar Tutorial Huruf {self.letter}")
        else:
            image_label.setText(f"🎬\nGambar Tutorial {self.letter}\n(Akan ditambahkan)")
        
        image_layout.addWidget(image_label, 0, Qt.AlignCenter)
        image_frame.setLayout(image_layout)
        scroll_layout.addWidget(image_frame)
        
        # Tutorial information
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border: 2px solid #57cc99;
                border-radius: 15px;
                padding: 20px;
                margin-top: 15px;
            }
        """)
        info_layout = QVBoxLayout()
        
        # Get tutorial text based on letter
        tutorial_text = self.get_tutorial_text()
        info_text = QLabel(tutorial_text)
        
        # Gunakan font config untuk text
        text_font_size = 14
        if hasattr(self.parent_app, 'font_config'):
            text_font_size = self.parent_app.font_config.get('tutorial_text', 14)
        
        info_text.setStyleSheet(f"""
            font-size: {text_font_size}px;
            color: #e6e6e6;
            line-height: 1.6;
        """)
        info_text.setWordWrap(True)
        info_text.setTextFormat(Qt.RichText)
        
        info_layout.addWidget(info_text)
        info_frame.setLayout(info_layout)
        scroll_layout.addWidget(info_frame)
        
        # Tips section
        tips_frame = QFrame()
        tips_frame.setStyleSheet("""
            QFrame {
                background-color: #1a5f7a;
                border: 2px solid #80ed99;
                border-radius: 15px;
                padding: 20px;
                margin-top: 15px;
            }
        """)
        tips_layout = QVBoxLayout()
        
        tips_title = QLabel("💡 Tips Praktik:")
        tips_title.setStyleSheet(f"""
            font-size: {text_font_size + 2}px;
            font-weight: bold;
            color: #80ed99;
            margin-bottom: 10px;
        """)
        tips_layout.addWidget(tips_title)
        
        tips_list = QTextEdit()
        tips_list.setReadOnly(True)
        tips_list.setHtml("""
            <ul style='color: #e6e6e6;'>
                <li>Pastikan pencahayaan cukup dari depan</li>
                <li>Jaga jarak 30-50 cm dari kamera</li>
                <li>Gunakan latar belakang kontras (tidak seragam dengan kulit)</li>
                <li>Jaga tangan tetap stabil selama 2-3 detik</li>
                <li>Posisikan tangan di tengah frame kamera</li>
                <li>Hindari gerakan tangan yang terlalu cepat</li>
            </ul>
        """)
        tips_list.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                color: #e6e6e6;
                font-size: {text_font_size}px;
            }}
        """)
        tips_list.setMaximumHeight(150)
        tips_layout.addWidget(tips_list)
        
        tips_frame.setLayout(tips_layout)
        scroll_layout.addWidget(tips_frame)
        
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout()
        
        close_btn = QPushButton("Tutup")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #57cc99;
                color: #1a1a2e;
                font-weight: bold;
                padding: 10px 30px;
                font-size: 16px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #80ed99;
            }
        """)
        
        practice_btn = QPushButton("Coba Praktik")
        practice_btn.clicked.connect(self.start_practice)
        practice_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9a3c;
                color: #1a1a2e;
                font-weight: bold;
                padding: 10px 30px;
                font-size: 16px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #ffcc29;
            }
        """)
        
        button_layout.addWidget(practice_btn)
        button_layout.addWidget(close_btn)
        button_frame.setLayout(button_layout)
        layout.addWidget(button_frame)
        
        self.setLayout(layout)
        
    def get_tutorial_text(self):
        """Get tutorial text for specific letter"""
        tutorials = {
        'A': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf A:</h3>
                <p>1. Kepalkan tangan Anda dengan erat<br>
                2. Letakkan ibu jari di sisi tangan (menempel pada jari telunjuk)<br>
                3. Pastikan semua jari terkepal dengan rapat<br>
                4. Arahkan telapak tangan menghadap ke depan<br>
                5. Posisikan tangan setinggi dada</p>
                """,
        'B': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf B:</h3>
                <p>1. Buka telapak tangan selebar mungkin<br>
                2. Rapatkan keempat jari (telunjuk hingga kelingking)<br>
                3. Tekuk ibu jari ke dalam telapak tangan<br>
                4. Arahkan telapak tangan ke depan<br>
                5. Jaga jari-jari tetap lurus dan rapat</p>
                """,
        'C': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf C:</h3>
                <p>1. Lengkungkan empat jari ke arah depan secara bersamaan<br>
                2. Lengkungkan ibu jari ke atas membentuk busur<br>
                3. Pastikan tangan membentuk lubang setengah lingkaran<br>
                4. Hadapkan telapak tangan ke arah samping (profil)<br>
                5. Jaga jarak antara ibu jari dan ujung jari lainnya</p>
                """,
        'D': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf D:</h3>
                <p>1. Angkat jari telunjuk lurus ke atas<br>
                2. Pertemukan ujung ibu jari dengan ujung jari tengah, manis, dan kelingking<br>
                3. Pastikan jari-jari tersebut membentuk lingkaran kecil di bawah telunjuk<br>
                4. Arahkan telapak tangan ke depan<br>
                5. Jaga jari telunjuk tetap tegak lurus</p>
                """,
        'E': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf E:</h3>
                <p>1. Tekuk keempat jari ke arah dalam telapak tangan<br>
                2. Lipat ibu jari ke depan telapak tangan (di bawah ujung jari lainnya)<br>
                3. Pastikan kuku jari-jari menyentuh bagian atas ibu jari<br>
                4. Posisi tangan seperti cakar yang merapat<br>
                5. Hadapkan telapak tangan ke depan</p>
                """,
        'F': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf F:</h3>
                <p>1. Pertemukan ujung jari telunjuk dengan ujung ibu jari (membentuk lingkaran)<br>
                2. Angkat tiga jari lainnya (tengah, manis, kelingking) lurus ke atas<br>
                3. Regangkan sedikit ketiga jari yang berdiri tersebut<br>
                4. Arahkan telapak tangan ke depan<br>
                5. Pastikan lingkaran antara telunjuk dan jempol terlihat jelas</p>
                """,
        'G': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf G:</h3>
                <p>1. Rentangkan jari telunjuk dan ibu jari secara sejajar<br>
                2. Tekuk jari lainnya (tengah, manis, kelingking) ke telapak tangan<br>
                3. Arahkan telunjuk dan jempol ke arah samping (seperti menunjuk sesuatu)<br>
                4. Pastikan telapak tangan menghadap ke arah Anda<br>
                5. Jari telunjuk dan jempol tidak bersentuhan</p>
                """,
        'H': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf H:</h3>
                <p>1. Rentangkan jari telunjuk dan jari tengah secara sejajar ke samping<br>
                2. Rapatkan kedua jari tersebut<br>
                3. Tekuk ibu jari dan jari lainnya ke telapak tangan<br>
                4. Arahkan telapak tangan menghadap ke arah Anda<br>
                5. Posisi jari horizontal ke arah samping</p>
                """,
        'I': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf I:</h3>
                <p>1. Angkat jari kelingking lurus ke atas<br>
                2. Kepalkan jari lainnya ke dalam telapak tangan<br>
                3. Tekuk ibu jari di atas jari tengah dan manis yang mengepal<br>
                4. Arahkan telapak tangan ke depan<br>
                5. Pastikan hanya kelingking yang berdiri tegak</p>
                """,
        'J': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf J:</h3>
                <p>1. Mulailah dengan posisi huruf I (kelingking tegak)<br>
                2. Gerakkan tangan Anda di udara membentuk gerakan "kait" atau huruf J<br>
                3. Ujung kelingking seolah-olah menggambar huruf J di udara<br>
                4. Akhiri gerakan dengan kelingking menghadap ke arah Anda<br>
                5. Pastikan gerakan mengalir lancar</p>
                """,
        'K': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf K:</h3>
                <p>1. Angkat jari telunjuk dan jari tengah ke atas (membentuk V)<br>
                2. Letakkan ujung ibu jari di tengah-tengah antara telunjuk dan jari tengah<br>
                3. Tekuk jari manis dan kelingking ke telapak tangan<br>
                4. Arahkan telapak tangan ke depan<br>
                5. Jari telunjuk harus tetap lurus ke atas</p>
                """,
        'L': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf L:</h3>
                <p>1. Angkat jari telunjuk lurus ke atas<br>
                2. Rentangkan ibu jari ke samping secara horizontal<br>
                3. Tekuk tiga jari lainnya ke telapak tangan<br>
                4. Pastikan sudut antara telunjuk dan jempol membentuk 90 derajat<br>
                5. Hadapkan telapak tangan ke depan</p>
                """,
        'M': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf M:</h3>
                <p>1. Kepalkan tangan Anda<br>
                2. Selipkan ibu jari di bawah jari telunjuk, tengah, dan manis<br>
                3. Biarkan ujung ibu jari muncul di antara jari manis dan kelingking<br>
                4. Pastikan telapak tangan menghadap ke depan<br>
                5. Tiga jari (telunjuk, tengah, manis) menutupi jempol</p>
                """,
        'N': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf N:</h3>
                <p>1. Kepalkan tangan Anda<br>
                2. Selipkan ibu jari di bawah jari telunjuk dan jari tengah<br>
                3. Biarkan ujung ibu jari muncul di antara jari tengah dan jari manis<br>
                4. Pastikan telapak tangan menghadap ke depan<br>
                5. Dua jari (telunjuk, tengah) menutupi jempol</p>
                """,
        'O': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf O:</h3>
                <p>1. Pertemukan semua ujung jari dengan ujung ibu jari<br>
                2. Pastikan bentuknya bulat sempurna seperti huruf O<br>
                3. Jaga agar ada lubang di tengah lingkaran jari tersebut<br>
                4. Hadapkan telapak tangan ke arah samping atau sedikit ke depan<br>
                5. Jangan menekuk jari terlalu tajam</p>
                """,
        'P': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf P:</h3>
                <p>1. Posisikan tangan seperti huruf K tetapi arahkan ke bawah<br>
                2. Jari telunjuk menunjuk lurus ke lantai<br>
                3. Jari tengah horizontal menjauh dari telapak tangan<br>
                4. Letakkan ibu jari di atas jari tengah<br>
                5. Hadapkan telapak tangan ke bawah</p>
                """,
        'Q': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf Q:</h3>
                <p>1. Posisikan tangan seperti huruf G tetapi arahkan ke bawah<br>
                2. Jari telunjuk dan ibu jari menunjuk ke arah lantai<br>
                3. Tekuk jari lainnya ke dalam telapak tangan<br>
                4. Biarkan ada sedikit jarak antara telunjuk dan jempol<br>
                5. Hadapkan telapak tangan ke arah tubuh</p>
                """,
        'R': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf R:</h3>
                <p>1. Angkat jari telunjuk dan jari tengah lurus ke atas<br>
                2. Silangkan jari tengah di belakang jari telunjuk<br>
                3. Tekuk ibu jari di atas jari manis dan kelingking yang mengepal<br>
                4. Pastikan kedua jari yang bersilangan tetap tegak<br>
                5. Hadapkan telapak tangan ke depan</p>
                """,
        'S': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf S:</h3>
                <p>1. Kepalkan tangan Anda dengan kuat<br>
                2. Letakkan ibu jari di depan keempat jari yang mengepal (di tengah)<br>
                3. Pastikan jempol tidak berada di samping (seperti huruf A)<br>
                4. Hadapkan telapak tangan ke depan<br>
                5. Tekan jempol ke arah jari-jari lainnya</p>
                """,
        'T': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf T:</h3>
                <p>1. Kepalkan tangan Anda<br>
                2. Selipkan ibu jari di bawah jari telunjuk<br>
                3. Biarkan ujung ibu jari muncul di antara jari telunjuk dan jari tengah<br>
                4. Pastikan telapak tangan menghadap ke depan<br>
                5. Hanya satu jari (telunjuk) yang menutupi jempol</p>
                """,
        'U': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf U:</h3>
                <p>1. Angkat jari telunjuk dan jari tengah lurus ke atas<br>
                2. Rapatkan kedua jari tersebut tanpa celah<br>
                3. Tekuk ibu jari di atas jari manis dan kelingking yang mengepal<br>
                4. Hadapkan telapak tangan ke depan<br>
                5. Pastikan jari tetap tegak dan lurus</p>
                """,
        'V': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf V:</h3>
                <p>1. Angkat jari telunjuk dan jari tengah ke atas (seperti tanda damai)<br>
                2. Regangkan kedua jari tersebut hingga membentuk huruf V<br>
                3. Tekuk ibu jari di atas jari manis dan kelingking yang mengepal<br>
                4. Hadapkan telapak tangan ke depan<br>
                5. Jari lainnya harus tertutup rapat</p>
                """,
        'W': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf W:</h3>
                <p>1. Angkat jari telunjuk, tengah, dan manis lurus ke atas<br>
                2. Regangkan ketiga jari tersebut sedikit membentuk huruf W<br>
                3. Pertemukan ujung ibu jari dengan ujung kelingking<br>
                4. Hadapkan telapak tangan ke depan<br>
                5. Jaga agar jari yang berdiri tetap stabil</p>
                """,
        'X': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf X:</h3>
                <p>1. Angkat jari telunjuk dan tekuk separuh membentuk kait (seperti kail)<br>
                2. Kepalkan jari lainnya ke dalam telapak tangan<br>
                3. Letakkan ibu jari di samping kepalan tangan<br>
                4. Hadapkan telapak tangan ke depan atau sedikit menyamping<br>
                5. Jari telunjuk harus terlihat melengkung</p>
                """,
        'Y': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf Y:</h3>
                <p>1. Rentangkan ibu jari dan jari kelingking sejauh mungkin<br>
                2. Tekuk jari telunjuk, tengah, dan manis ke arah telapak tangan<br>
                3. Pastikan telapak tangan menghadap ke depan<br>
                4. Posisi ini menyerupai bentuk telepon atau tanduk<br>
                5. Jaga jari jempol dan kelingking tetap lurus</p>
                """,
        'Z': """
                <h3 style='color: #57cc99;'>Cara Membuat Isyarat Huruf Z:</h3>
                <p>1. Angkat jari telunjuk lurus ke depan<br>
                2. Kepalkan jari lainnya ke dalam telapak tangan<br>
                3. Gerakkan jari telunjuk di udara membentuk pola zigzag (huruf Z)<br>
                4. Gerakan terdiri dari garis horizontal, diagonal, lalu horizontal lagi<br>
                5. Gunakan pergelangan tangan untuk mengontrol gerakan</p>
                """,
    }
        
        return tutorials.get(self.letter, 
            f"<h3 style='color: #57cc99;'>Tutorial Huruf {self.letter}:</h3>"
            f"<p>Informasi detail untuk huruf {self.letter} akan segera ditambahkan.</p>")
    
    def start_practice(self):
        """Switch to detection tab for practice"""
        self.accept()
        if self.parent():
            self.parent().tab_widget.setCurrentIndex(0)  # Switch to detection tab
            if not self.parent().is_running:
                self.parent().toggle_detection()
            QMessageBox.information(self, "Mode Praktik", 
                f"Sekarang praktikkan huruf {self.letter} di depan kamera!\n\n"
                f"Tips: {self.get_practice_tip()}")

    def get_practice_tip(self):
        """Get specific practice tip for the letter"""
        tips = {
            'A': "Pastikan tangan benar-benar terkepal dan ibu jari terlihat jelas.",
            'B': "Jaga telapak tangan rata dan jari-jari benar-benar rapat.",
            # ... (tips untuk huruf lainnya)
        }
        return tips.get(self.letter, "Coba variasikan posisi untuk mendapatkan akurasi terbaik.")

class MultimediaSignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.mp_hands = None
        self.hands = None
        self.cap = None
        self.is_running = False
        self.prediction_history = []
        self.accuracy_history = []
        self.session_data = []
        self.session_start_time = None
        self.frame_count = 0
        self.last_fps_update = datetime.datetime.now()
        
        # Font configuration
        self.font_config = self.get_font_config()
        
        # Konfigurasi
        class Config:
            BASE_DIR = project_root
            DATA_DIR = os.path.join(project_root, 'data')
            MODELS_DIR = os.path.join(DATA_DIR, 'models')
            RESULTS_DIR = os.path.join(project_root, 'results')
            PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
            NUM_FEATURES = 63
            LETTERS = [chr(i) for i in range(65, 91)]
        
        self.config = Config()
        
        self.init_ui()
        self.load_model()
    
    def get_font_config(self):
        """Get font configuration based on screen size"""
        # Get screen size
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        width = screen_size.width()
        
        if width < 1366:  # Laptop kecil
            return {
                "prediction": 100,     # Huruf prediksi
                "main_title": 24,      # Judul utama
                "subtitle": 12,        # Subjudul
                "tab_title": 16,       # Judul tab
                "button": 24,          # Tombol huruf
                "history": 10,         # Riwayat
                "metric": 20,          # Metrik analytics
                "tutorial_header": 24, # Header tutorial
                "tutorial_text": 13    # Teks tutorial
            }
        elif width < 1920:  # Laptop standar
            return {
                "prediction": 100,
                "main_title": 28,
                "subtitle": 14,
                "tab_title": 18,
                "button": 28,
                "history": 12,
                "metric": 22,
                "tutorial_header": 28,
                "tutorial_text": 14
            }
        else:  # Monitor besar
            return {
                "prediction": 100,
                "main_title": 32,
                "subtitle": 16,
                "tab_title": 20,
                "button": 32,
                "history": 14,
                "metric": 24,
                "tutorial_header": 32,
                "tutorial_text": 24
            }
    
    # ==================== CORE METHODS ====================
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            if not os.path.exists(self.config.MODELS_DIR):
                self.show_error("Model Directory Not Found", 
                    f"Directory tidak ditemukan: {self.config.MODELS_DIR}\n\n"
                    "Silakan jalankan train_coordinate_model.py terlebih dahulu.")
                return False
            
            # Find model file
            model_path = None
            for file in ['coordinate_model.h5', 'coordinate_model_final.h5']:
                path = os.path.join(self.config.MODELS_DIR, file)
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                self.show_error("Model Not Found", 
                    "File model tidak ditemukan!\n\n"
                    "Silakan train model terlebih dahulu:\n"
                    "python train_coordinate_model.py")
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.config.MODELS_DIR, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                dummy_data = np.zeros((1, self.config.NUM_FEATURES))
                self.scaler.fit(dummy_data)
            
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.status_label.setText("✅ Model berhasil dimuat! Klik 'Mulai Deteksi' untuk memulai.")
            return True
            
        except Exception as e:
            self.show_error("Load Model Error", f"Gagal memuat model:\n{str(e)}")
            return False
    
    def toggle_detection(self):
        """Start/stop detection"""
        if not self.is_running:
            if self.model is None:
                self.show_error("Model Not Loaded", "Model belum dimuat!")
                return
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Camera Error", "Tidak dapat mengakses kamera!")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.start_btn.setText("⏸ Berhenti Deteksi")
            self.timer.start(30)
            self.status_label.setText("🔴 Deteksi aktif - Tunjukkan tangan dengan isyarat huruf")
            
            # Reset analytics
            self.prediction_history = []
            self.accuracy_history = []
            self.session_data = []
            self.session_start_time = datetime.datetime.now()
            self.frame_count = 0
            
        else:
            self.is_running = False
            self.start_btn.setText("▶ Mulai Deteksi")
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.status_label.setText("✅ Deteksi dihentikan")
    
    def update_frame(self):
        """Update video frame and prediction"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        start_time = datetime.datetime.now()
        self.frame_count += 1
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            prediction = "?"
            confidence = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    if len(landmarks) == self.config.NUM_FEATURES:
                        try:
                            processed = self.scaler.transform([landmarks])
                            predictions = self.model.predict(processed, verbose=0)
                            pred_idx = np.argmax(predictions[0])
                            confidence = float(np.max(predictions[0]))
                            
                            if pred_idx < len(self.config.LETTERS):
                                prediction = self.config.LETTERS[pred_idx]
                                
                                # Record session data
                                self.session_data.append({
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'landmarks': landmarks[:10]
                                })
                                
                                # Add to history
                                self.prediction_history.append(prediction)
                                self.accuracy_history.append(confidence)
                        except Exception as e:
                            print(f"Prediction error: {e}")
            
            # Update UI
            self.prediction_label.setText(prediction)
            confidence_percent = int(confidence * 100)
            self.confidence_bar.setValue(confidence_percent)
            #self.confidence_label.setText(f"{confidence_percent}%")
            
            # Change color based on confidence
            if confidence > 0.8:
                color = "#57cc99"  # Green
            elif confidence > 0.6:
                color = "#ff9a3c"  # Orange
            else:
                color = "#ff6b6b"  # Red
            
            self.prediction_label.setStyleSheet(f"""
                font-size: {self.font_config['prediction']}px;
                font-weight: bold;
                color: {color};
            """)
            
            # Process frame for display
            frame = cv2.flip(frame, 1)
            
            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
            
            # Add overlay text
            cv2.putText(frame, f"Prediksi: {prediction}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 204, 153), 3)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (87, 204, 153), 2)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (500, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Convert to QImage
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            # Display
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pixmap)
            
            # Add to history list
            if confidence > 0.7 and prediction != "?":
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                item_text = f"[{timestamp}] {prediction} - {confidence*100:.1f}%"
                self.history_list.insertItem(0, item_text)
                
                if self.history_list.count() > 20:
                    self.history_list.takeItem(20)
            
            # Calculate FPS
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Update FPS setiap 0.5 detik
            current_time = datetime.datetime.now()
            if (current_time - self.last_fps_update).total_seconds() > 0.5:
                self.fps_indicator.setText(f"FPS: {fps:.1f}")
                self.last_fps_update = current_time
            
            # Update status bar
            self.status_label.setText(
                f"🔴 Deteksi aktif | Prediksi: {prediction} | "
                f"Confidence: {confidence*100:.1f}%"
            )
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
    
    def update_analytics(self):
        """Update analytics displays"""
        try:
            # Update performance metrics
            if hasattr(self, 'session_data') and self.session_data:
                total_detections = len(self.session_data)
                avg_confidence = np.mean([d['confidence'] for d in self.session_data]) * 100 if self.session_data else 0
                max_confidence = np.max([d['confidence'] for d in self.session_data]) * 100 if self.session_data else 0
                
                # Calculate session duration
                session_duration = "0:00"
                if self.session_start_time:
                    duration = datetime.datetime.now() - self.session_start_time
                    minutes = int(duration.total_seconds() // 60)
                    seconds = int(duration.total_seconds() % 60)
                    session_duration = f"{minutes}:{seconds:02d}"
                
                # Update metrics in analytics tab
                if hasattr(self, 'detection_count_label'):
                    self.detection_count_label.setText(f"Total Deteksi: {total_detections}")
                if hasattr(self, 'avg_confidence_label'):
                    self.avg_confidence_label.setText(f"Rata-rata Confidence: {avg_confidence:.1f}%")
                if hasattr(self, 'max_confidence_label'):
                    self.max_confidence_label.setText(f"Akurasi Tertinggi: {max_confidence:.1f}%")
                if hasattr(self, 'session_duration_label'):
                    self.session_duration_label.setText(f"Sesi Aktif: {session_duration}")
            
            # Update accuracy chart
            self.update_accuracy_chart()
            
            # Update distribution chart
            self.update_distribution_chart()
            
        except Exception as e:
            print(f"Error in update_analytics: {e}")
    
    def update_accuracy_chart(self):
        """Update accuracy trend chart"""
        if len(self.accuracy_history) > 1:
            self.acc_ax.clear()
            
            x = range(len(self.accuracy_history))
            y = [acc * 100 for acc in self.accuracy_history]
            
            self.acc_ax.plot(x, y, color='#57cc99', linewidth=2, marker='o', markersize=4)
            self.acc_ax.fill_between(x, y, alpha=0.3, color='#57cc99')
            
            self.acc_ax.set_facecolor('#0f3460')
            self.acc_ax.tick_params(colors='#e6e6e6')
            self.acc_ax.set_xlabel('Deteksi ke-', color='#80ed99')
            self.acc_ax.set_ylabel('Akurasi (%)', color='#80ed99')
            self.acc_ax.set_title('Perkembangan Akurasi Deteksi', color='#57cc99')
            self.acc_ax.grid(True, alpha=0.3)
            self.acc_ax.set_ylim(0, 100)
            
            self.acc_canvas.draw()
    
    def update_distribution_chart(self):
        """Update letter distribution chart"""
        if self.prediction_history:
            from collections import Counter
            counts = Counter(self.prediction_history)
            
            letters = list(counts.keys())
            frequencies = list(counts.values())
            
            self.dist_ax.clear()
            
            colors = ['#57cc99', '#80ed99', '#ff9a3c', '#ffcc29', '#6a67ce', '#ff6b6b']
            bars = self.dist_ax.bar(letters, frequencies, color=colors[:len(letters)])
            
            # Add value labels on bars
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                self.dist_ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{freq}', ha='center', va='bottom', color='white', fontweight='bold')
            
            self.dist_ax.set_facecolor('#0f3460')
            self.dist_ax.tick_params(colors='#e6e6e6')
            self.dist_ax.set_xlabel('Huruf', color='#80ed99')
            self.dist_ax.set_ylabel('Frekuensi', color='#80ed99')
            self.dist_ax.set_title('Distribusi Huruf Terdeteksi', color='#6a67ce')
            
            self.dist_canvas.draw()
    
    def update_clock(self):
        """Update time in status bar"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.time_indicator.setText(current_time)
    
    def update_memory_usage(self):
        """Update memory usage in status bar"""
        try:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
            self.mem_indicator.setText(f"RAM: {memory_usage:.1f} MB")
        except:
            pass
    
    def capture_frame(self):
        """Capture screenshot"""
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                captures_dir = os.path.join(self.config.BASE_DIR, 'captures')
                os.makedirs(captures_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(captures_dir, f'capture_{timestamp}.jpg')
                
                # Add watermark
                cv2.putText(frame, "Penerjemah Bahasa Isyarat", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 204, 153), 2)
                cv2.putText(frame, timestamp, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                cv2.imwrite(filename, frame)
                
                QMessageBox.information(self, "Screenshot Tersimpan",
                    f"Screenshot berhasil disimpan:\n{filename}")
            else:
                QMessageBox.warning(self, "Gagal", "Tidak dapat mengambil screenshot")
        else:
            QMessageBox.warning(self, "Perhatian", 
                "Mulai deteksi terlebih dahulu sebelum mengambil screenshot")
    
    def show_tutorial(self):
        """Show tutorial dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🎬 Mode Tutorial")
        dialog.setFixedSize(600, 400)
        dialog.setStyleSheet(APP_STYLE)
        
        layout = QVBoxLayout()
        
        title = QLabel("MODE TUTORIAL - BELAJAR BAHASA ISYARAT")
        title.setStyleSheet(f"""
            font-size: {self.font_config['tab_title'] + 2}px;
            font-weight: bold;
            color: #57cc99;
            padding: 10px;
            border-bottom: 2px solid #57cc99;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        content = QTextEdit()
        content.setReadOnly(True)
        content.setHtml("""
        <div style='color: #e6e6e6; padding: 20px;'>
            <h2 style='color: #57cc99;'>Selamat Datang di Mode Tutorial!</h2>
            
            <p>Mode ini membantu Anda belajar bahasa isyarat dengan cara interaktif:</p>
            
            <h3 style='color: #80ed99;'>🎯 Cara Penggunaan:</h3>
            <ol>
                <li>Pilih huruf yang ingin dipelajari di tab "Mode Pembelajaran"</li>
                <li>Lihat panduan visual cara membuat isyarat</li>
                <li>Praktikkan di depan kamera</li>
                <li>Pantau akurasi deteksi di tab "Analisis Performa"</li>
            </ol>
            
            <h3 style='color: #80ed99;'>💡 Tips Belajar:</h3>
            <ul>
                <li>Mulai dengan huruf-huruf dasar (A, B, C)</li>
                <li>Ulangi setiap isyarat beberapa kali</li>
                <li>Gunakan pencahayaan yang baik</li>
                <li>Latihan konsisten meningkatkan akurasi</li>
            </ul>
            
            <div style='background-color: #0f3460; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                <p style='color: #57cc99; text-align: center;'>
                    <strong>Selamat belajar! 🎓</strong><br>
                    Praktek teratur adalah kunci keberhasilan
                </p>
            </div>
        </div>
        """)
        content.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                font-size: {self.font_config['tutorial_text']}px;
            }}
        """)
        layout.addWidget(content)
        
        close_btn = QPushButton("Tutup")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #57cc99;
                color: #1a1a2e;
                font-weight: bold;
                padding: 10px;
                font-size: 16px;
            }
        """)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_letter_tutorial(self, letter):
        """Show tutorial dialog for specific letter"""
        dialog = TutorialDialog(letter, self)  # Pass self as parent
        dialog.exec_()
    
    def show_error(self, title, message):
        """Show error message dialog"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet(APP_STYLE)
        msg.exec_()
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_running:
            self.toggle_detection()
        
        # Save session data
        if self.session_data:
            sessions_dir = os.path.join(self.config.BASE_DIR, 'sessions')
            os.makedirs(sessions_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(sessions_dir, f'session_{timestamp}.json')
            
            with open(filename, 'w') as f:
                json.dump({
                    'session_end': datetime.datetime.now().isoformat(),
                    'total_predictions': len(self.session_data),
                    'average_confidence': np.mean([d['confidence'] for d in self.session_data]) if self.session_data else 0,
                    'data': self.session_data[:100]  # Save only first 100 records
                }, f, indent=2)
        
        event.accept()
    
    # ==================== UI CREATION METHODS ====================
    
    def init_ui(self):
        """Initialize multimedia user interface optimized for fullscreen"""
        self.setWindowTitle("🖐️ Inovasi Multimedia: Penerjemah Bahasa Isyarat Indonesia")
        
        # Set untuk fullscreen
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.setGeometry(screen_geometry)
        
        # Atur window untuk bisa maximize dan fullscreen
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | 
                          Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        self.setStyleSheet(APP_STYLE)
        
        # Central widget dengan background gradient untuk fullscreen
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            background: qlineargradient(
                spread:pad, x1:0, y1:0, x2:1, y2:1,
                stop:0 #1a1a2e, stop:1 #16213e
            );
        """)
        self.setCentralWidget(central_widget)
        
        # Main layout dengan stretch untuk fullscreen
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        
        # Header dengan judul dan informasi
        header_frame = self.create_header()
        main_layout.addWidget(header_frame)
        
        # Content area dengan tab widget yang fleksibel
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 3px solid #57cc99;
                border-radius: 15px;
                padding: 10px;
                background-color: #16213e;
            }
            QTabBar::tab {
                padding: 12px 24px;
                margin-right: 4px;
            }
        """)
        
        # Tab 1: Real-time Detection
        detection_tab = self.create_detection_tab()
        self.tab_widget.addTab(detection_tab, "🎥 Deteksi Real-time")
        
        # Tab 2: Learning Mode
        learning_tab = self.create_learning_tab()
        self.tab_widget.addTab(learning_tab, "📚 Mode Pembelajaran")
        
        # Tab 3: Performance Analytics
        analytics_tab = self.create_analytics_tab()
        self.tab_widget.addTab(analytics_tab, "📊 Analisis Performa")
        
        # Tab 4: About & Help
        about_tab = self.create_about_tab()
        self.tab_widget.addTab(about_tab, "ℹ️ Tentang")
        
        main_layout.addWidget(self.tab_widget, 1)  # Stretch factor = 1
        
        # Status bar dengan informasi lebih detail
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #0f3460;
                color: #e6e6e6;
                padding: 5px;
                border-top: 2px solid #57cc99;
            }
        """)
        
        # Tambah widget permanen di status bar
        self.status_label = QLabel("✅ Aplikasi siap digunakan. Tekan 'Mulai Deteksi' untuk memulai.")
        self.status_bar.addWidget(self.status_label, 1)
        
        # FPS indicator
        self.fps_indicator = QLabel("FPS: 0.0")
        self.fps_indicator.setStyleSheet(f"font-size: {self.font_config['subtitle']}px;")
        self.status_bar.addPermanentWidget(self.fps_indicator)
        
        # Memory indicator
        self.mem_indicator = QLabel("RAM: -")
        self.mem_indicator.setStyleSheet(f"font-size: {self.font_config['subtitle']}px;")
        self.status_bar.addPermanentWidget(self.mem_indicator)
        
        # Time indicator
        self.time_indicator = QLabel(datetime.datetime.now().strftime("%H:%M:%S"))
        self.time_indicator.setStyleSheet(f"font-size: {self.font_config['subtitle']}px;")
        self.status_bar.addPermanentWidget(self.time_indicator)
        
        self.setStatusBar(self.status_bar)
        
        # Timer untuk video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Timer untuk update analytics
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.update_analytics)
        self.analytics_timer.start(2000)
        
        # Timer untuk update waktu
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        
        # Timer untuk update memory usage
        self.mem_timer = QTimer()
        self.mem_timer.timeout.connect(self.update_memory_usage)
        self.mem_timer.start(5000)
    
    def create_header(self):
        """Create application header optimized for fullscreen"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border-radius: 15px;
                border: 3px solid #57cc99;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        
        # Logo/Icon area
        logo_label = QLabel("🖐️")
        logo_label.setStyleSheet("font-size: 48px; min-width: 60px;")
        logo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo_label)
        
        # Title and description dengan layout fleksibel
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        title_label = QLabel("Penerjemah Bahasa Isyarat Indonesia")
        title_label.setStyleSheet(f"""
            font-size: {self.font_config['main_title']}px;
            font-weight: bold;
            color: #57cc99;
            padding: 5px;
        """)
        
        subtitle_label = QLabel("Inovasi Multimedia - Sistem Deteksi Real-time dengan MediaPipe & AI")
        subtitle_label.setStyleSheet(f"""
            font-size: {self.font_config['subtitle']}px;
            color: #80ed99;
            padding: 2px;
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout, 1)  # Stretch factor
        
        # Quick stats dengan layout responsif
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 10px;
                padding: 10px;
                min-width: 300px;
            }
        """)
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(10)
        
        stats = [
            ("📊", "Akurasi", "95%"),
            ("⚡", "FPS", "30"),
            ("🕒", "Waktu", datetime.datetime.now().strftime("%H:%M")),
            ("🔤", "Huruf", "A-Z")
        ]
        
        for icon, label, value in stats:
            stat_widget = self.create_stat_widget(icon, label, value)
            stats_layout.addWidget(stat_widget)
        
        stats_frame.setLayout(stats_layout)
        header_layout.addWidget(stats_frame)
        
        header_frame.setLayout(header_layout)
        return header_frame
    
    def create_stat_widget(self, icon, label, value):
        """Create a responsive statistic widget"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border-radius: 8px;
                padding: 8px;
                margin: 2px;
                min-width: 70px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(2)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setAlignment(Qt.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: {self.font_config['metric']}px; font-weight: bold; color: #57cc99;")
        value_label.setAlignment(Qt.AlignCenter)
        
        label_label = QLabel(label)
        label_label.setStyleSheet(f"font-size: {self.font_config['history']}px; color: #e6e6e6;")
        label_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(icon_label)
        layout.addWidget(value_label)
        layout.addWidget(label_label)
        widget.setLayout(layout)
        
        return widget
    
    def create_detection_tab(self):
        """Create real-time detection tab optimized for fullscreen"""
        tab = QWidget()
        layout = QHBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel - Video and controls
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f3460, stop:1 #16213e
                );
                border: 3px solid #57cc99;
                border-radius: 15px;
            }
        """)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Video display dengan ukuran responsif
        video_container = QFrame()
        video_container.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border-radius: 15px;
                border: 3px solid #57cc99;
            }
        """)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border-radius: 10px;
                background-color: #000000;
            }
        """)
        video_layout.addWidget(self.video_label, 1)  # Stretch factor
        
        video_container.setLayout(video_layout)
        left_layout.addWidget(video_container, 3)  # Lebih besar proportion
        
        # Control buttons dengan layout responsif
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        
        self.start_btn = QPushButton("▶ Mulai Deteksi")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_btn.setMinimumHeight(45)
        
        self.capture_btn = QPushButton("📸 Ambil Screenshot")
        self.capture_btn.setObjectName("captureBtn")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setMinimumHeight(45)
        
        self.tutorial_btn = QPushButton("🎬 Mode Tutorial")
        self.tutorial_btn.setObjectName("tutorialBtn")
        self.tutorial_btn.clicked.connect(self.show_tutorial)
        self.tutorial_btn.setMinimumHeight(45)
        
        control_layout.addWidget(self.start_btn, 1)
        control_layout.addWidget(self.capture_btn, 1)
        control_layout.addWidget(self.tutorial_btn, 1)
        control_frame.setLayout(control_layout)
        left_layout.addWidget(control_frame)
        
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 60)  # 60% width
        
        # Right panel - Results and info
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(50)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Prediction display
        prediction_frame = QFrame()
        prediction_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f3460, stop:1 #1a5f7a
                );
                border-radius: 15px;
                border: 3px solid #57cc99;
            }
        """)
        prediction_layout = QVBoxLayout()
        prediction_layout.setSpacing(10)
        prediction_layout.setContentsMargins(15, 15, 15, 15)
        
        prediction_title = QLabel("HURUF TERDETEKSI")
        prediction_title.setStyleSheet(f"""
            font-size: {self.font_config['tab_title']}px;
            font-weight: bold;
            color: #80ed99;
        """)
        prediction_title.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(prediction_title)
        
        self.prediction_label = QLabel("?")
        self.prediction_label.setStyleSheet(f"""
            font-size: {self.font_config['prediction']}px;
            font-weight: bold;
            color: #57cc99;
        """)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.prediction_label, 1)
        
        # Confidence visualization
        confidence_frame = QFrame()
        confidence_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 52, 96, 150);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        confidence_layout = QVBoxLayout()
        confidence_layout.setSpacing(5)
        
        confidence_header = QLabel("TINGKAT KEPERCAYAAN")
        confidence_header.setStyleSheet(f"""
            font-size: 12px;
            font-weight: bold;
            color: #e6e6e6;
            border-radius: 10px
        """)
        confidence_header.setAlignment(Qt.AlignCenter)
        confidence_layout.addWidget(confidence_header)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%p%")
        self.confidence_bar.setMinimumHeight(25)
        confidence_layout.addWidget(self.confidence_bar)
        
        confidence_frame.setLayout(confidence_layout)
        prediction_layout.addWidget(confidence_frame)
        
        prediction_frame.setLayout(prediction_layout)
        right_layout.addWidget(prediction_frame, 40)  # 40% height
        
        # History log dengan scroll
        history_frame = QFrame()
        history_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        history_layout = QVBoxLayout()
        history_layout.setSpacing(5)
        
        history_title = QLabel("📜 RIWAYAT DETEKSI")
        history_title.setStyleSheet(f"""
            font-size: {self.font_config['tab_title']}px;
            font-weight: bold;
            color: #57cc99;
            padding: 5px;
            border-bottom: 2px solid #57cc99;
        """)
        history_layout.addWidget(history_title)
        
        self.history_list = QListWidget()
        self.history_list.setStyleSheet(f"""
            QListWidget {{
                background-color: #1a1a2e;
                border: 2px solid #0f3460;
                border-radius: 10px;
                padding: 5px;
                font-size: {self.font_config['history']}px;
            }}
            QListWidget::item {{
                padding: 8px;
                margin: 2px;
                border-radius: 5px;
                background-color: #0f3460;
                border: 1px solid #1a5f7a;
            }}
            QListWidget::item:hover {{
                background-color: #1a5f7a;
                border: 1px solid #57cc99;
            }}
        """)
        self.history_list.setAlternatingRowColors(True)
        history_layout.addWidget(self.history_list, 1)  # Stretch
        
        history_frame.setLayout(history_layout)
        right_layout.addWidget(history_frame, 60)  # 60% height
        
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, 40)  # 40% width
        
        tab.setLayout(layout)
        return tab
    
    def create_learning_tab(self):
        """Create learning/tutorial tab dengan grid huruf yang lebih rapi"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("📚 MODE PEMBELAJARAN BAHASA ISYARAT")
        title.setStyleSheet(f"""
            font-size: {self.font_config['main_title']}px;
            font-weight: bold;
            color: #57cc99;
            padding: 15px;
            border-bottom: 3px solid #57cc99;
            background-color: #0f3460;
            border-radius: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Alphabet grid dengan scroll area untuk responsif
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)
        
        alphabet_container = QWidget()
        alphabet_layout = QVBoxLayout()
        
        # Instruction
        instruction = QLabel("Pilih huruf untuk melihat tutorial detail:")
        instruction.setStyleSheet(f"""
            font-size: {self.font_config['tab_title']}px;
            color: #80ed99;
            padding: 10px;
            font-weight: bold;
        """)
        instruction.setAlignment(Qt.AlignCenter)
        alphabet_layout.addWidget(instruction)
        
        # Alphabet grid dengan 6 kolom
        alphabet_grid = QFrame()
        alphabet_grid_layout = QGridLayout()
        alphabet_grid_layout.setSpacing(15)
        alphabet_grid_layout.setContentsMargins(30, 20, 30, 20)
        
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Buat dictionary untuk warna berdasarkan grup huruf
        color_groups = {
            0: "#57cc99",  # A-F
            1: "#80ed99",  # G-L
            2: "#ff9a3c",  # M-R
            3: "#6a67ce",  # S-X
            4: "#ff6b6b"   # Y-Z
        }
        
        for i, letter in enumerate(letters):
            btn = QPushButton(letter)
            btn.setFixedSize(80, 80)
            
            # Tentukan warna berdasarkan grup
            group_idx = i // 6
            color = color_groups.get(group_idx, "#57cc99")
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: {self.font_config['button']}px;
                    font-weight: bold;
                    background-color: #0f3460;
                    color: {color};
                    border-radius: 15px;
                    border: 3px solid {color};
                }}
                QPushButton:hover {{
                    background-color: #1a5f7a;
                    border-width: 4px;
                    font-size: {self.font_config['button'] + 4}px;
                }}
                QPushButton:pressed {{
                    background-color: {color};
                    color: #1a1a2e;
                }}
            """)
            
            # Tambah tooltip
            btn.setToolTip(f"Klik untuk melihat tutorial huruf {letter}")
            
            btn.clicked.connect(lambda checked, l=letter: self.show_letter_tutorial(l))
            
            row = i // 6
            col = i % 6
            alphabet_grid_layout.addWidget(btn, row, col)
        
        alphabet_grid.setLayout(alphabet_grid_layout)
        alphabet_layout.addWidget(alphabet_grid)
        
        # Quick info panel
        quick_info = QLabel("💡 Tips: Klik huruf di atas untuk membuka window tutorial dengan gambar panduan")
        quick_info.setStyleSheet(f"""
            font-size: {self.font_config['subtitle']}px;
            color: #ffcc29;
            padding: 15px;
            background-color: #1a5f7a;
            border-radius: 10px;
            border-left: 5px solid #ffcc29;
        """)
        quick_info.setWordWrap(True)
        alphabet_layout.addWidget(quick_info)
        
        alphabet_container.setLayout(alphabet_layout)
        scroll_area.setWidget(alphabet_container)
        layout.addWidget(scroll_area, 1)  # Stretch factor
        
        tab.setLayout(layout)
        return tab
    
    def create_analytics_tab(self):
        """Create performance analytics tab untuk fullscreen"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("📊 ANALISIS PERFORMA SISTEM")
        title.setStyleSheet(f"""
            font-size: {self.font_config['main_title']}px;
            font-weight: bold;
            color: #57cc99;
            padding: 15px;
            border-bottom: 3px solid #57cc99;
            background-color: #0f3460;
            border-radius: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Analytics grid dengan stretch untuk fullscreen
        grid_widget = QWidget()
        grid_layout = QVBoxLayout()
        
        # First row: Accuracy and Performance
        first_row = QHBoxLayout()
        first_row.setSpacing(20)
        
        # Accuracy chart
        acc_frame = QFrame()
        acc_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 3px solid #57cc99;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        acc_layout = QVBoxLayout()
        
        acc_title = QLabel("📈 TREN AKURASI DETEKSI")
        acc_title.setStyleSheet(f"font-size: {self.font_config['tab_title']}px; font-weight: bold; color: #80ed99; padding: 5px;")
        acc_title.setAlignment(Qt.AlignCenter)
        acc_layout.addWidget(acc_title)
        
        # Matplotlib figure for accuracy chart
        self.acc_figure = Figure(figsize=(8, 4), facecolor='#1a1a2e', dpi=100)
        self.acc_canvas = FigureCanvas(self.acc_figure)
        self.acc_ax = self.acc_figure.add_subplot(111)
        self.acc_ax.set_facecolor('#0f3460')
        
        # Style the chart
        self.acc_ax.tick_params(colors='#e6e6e6', labelsize=10)
        self.acc_ax.set_xlabel('Waktu Deteksi', color='#80ed99', fontsize=11)
        self.acc_ax.set_ylabel('Akurasi (%)', color='#80ed99', fontsize=11)
        self.acc_ax.set_title('Perkembangan Akurasi Deteksi', color='#57cc99', fontsize=14, fontweight='bold')
        self.acc_ax.grid(True, alpha=0.3, linestyle='--', color='#57cc99')
        
        # Set axis limits
        self.acc_ax.set_ylim(0, 100)
        
        acc_layout.addWidget(self.acc_canvas)
        acc_frame.setLayout(acc_layout)
        first_row.addWidget(acc_frame, 2)  # 2/3 width
        
        # Performance metrics
        perf_frame = QFrame()
        perf_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 3px solid #ff9a3c;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        perf_layout = QVBoxLayout()
        perf_layout.setSpacing(15)
        
        perf_title = QLabel("⚡ METRIK KINERJA SISTEM")
        perf_title.setStyleSheet(f"font-size: {self.font_config['tab_title']}px; font-weight: bold; color: #ff9a3c; padding: 5px;")
        perf_title.setAlignment(Qt.AlignCenter)
        perf_layout.addWidget(perf_title)
        
        # Metrics dengan layout grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(15)
        
        # Create metric widgets
        metrics = [
            ("FPS", "0.0", "#57cc99", "Kecepatan frame per detik"),
            ("Latensi", "0 ms", "#80ed99", "Waktu pemrosesan per frame"),
            ("Total Deteksi", "0", "#ff9a3c", "Jumlah total deteksi"),
            ("Rata-rata Confidence", "0%", "#6a67ce", "Tingkat kepercayaan rata-rata"),
            ("Akurasi Tertinggi", "0%", "#ff6b6b", "Akurasi tertinggi yang dicapai"),
            ("Sesi Aktif", "0:00", "#ffcc29", "Durasi sesi deteksi")
        ]
        
        self.metric_labels = {}
        
        for i, (label, value, color, tooltip) in enumerate(metrics):
            metric_widget = self.create_metric_widget(label, value, color)
            metric_widget.setToolTip(tooltip)
            
            # Store reference to update later
            if label == "Total Deteksi":
                self.detection_count_label = metric_widget.findChild(QLabel, "value")
            elif label == "Rata-rata Confidence":
                self.avg_confidence_label = metric_widget.findChild(QLabel, "value")
            elif label == "Akurasi Tertinggi":
                self.max_confidence_label = metric_widget.findChild(QLabel, "value")
            elif label == "Sesi Aktif":
                self.session_duration_label = metric_widget.findChild(QLabel, "value")
            
            row = i // 2
            col = i % 2
            metrics_grid.addWidget(metric_widget, row, col)
        
        perf_layout.addLayout(metrics_grid)
        perf_frame.setLayout(perf_layout)
        first_row.addWidget(perf_frame, 1)  # 1/3 width
        
        grid_layout.addLayout(first_row)
        
        # Second row: Distribution chart (full width)
        dist_frame = QFrame()
        dist_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 3px solid #6a67ce;
                border-radius: 15px;
                padding: 20px;
                margin-top: 10px;
            }
        """)
        dist_layout = QVBoxLayout()
        
        dist_title = QLabel("🔤 DISTRIBUSI FREKUENSI HURUF TERDETEKSI")
        dist_title.setStyleSheet(f"font-size: {self.font_config['tab_title']}px; font-weight: bold; color: #6a67ce; padding: 5px;")
        dist_title.setAlignment(Qt.AlignCenter)
        dist_layout.addWidget(dist_title)
        
        # Distribution chart
        self.dist_figure = Figure(figsize=(10, 4), facecolor='#1a1a2e', dpi=100)
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_ax = self.dist_figure.add_subplot(111)
        self.dist_ax.set_facecolor('#0f3460')
        
        # Style the distribution chart
        self.dist_ax.tick_params(colors='#e6e6e6', labelsize=10)
        self.dist_ax.set_xlabel('Huruf', color='#80ed99', fontsize=11)
        self.dist_ax.set_ylabel('Frekuensi', color='#80ed99', fontsize=11)
        self.dist_ax.set_title('Distribusi Huruf yang Terdeteksi', color='#6a67ce', fontsize=14, fontweight='bold')
        self.dist_ax.grid(True, alpha=0.3, linestyle='--', color='#6a67ce')
        
        dist_layout.addWidget(self.dist_canvas)
        dist_frame.setLayout(dist_layout)
        grid_layout.addWidget(dist_frame)
        
        grid_widget.setLayout(grid_layout)
        
        # Add to scroll area untuk fullscreen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(grid_widget)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        layout.addWidget(scroll_area, 1)
        tab.setLayout(layout)
        return tab
    
    def create_metric_widget(self, label, value, color):
        """Create a metric display widget"""
        widget = QFrame()
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: #0f3460;
                border-radius: 10px;
                border: 2px solid {color};
                padding: 15px;
            }}
        """)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"""
            font-size: {self.font_config['subtitle']}px;
            color: #e6e6e6;
            font-weight: bold;
        """)
        label_widget.setAlignment(Qt.AlignCenter)
        label_widget.setObjectName("label")
        
        value_widget = QLabel(value)
        value_widget.setStyleSheet(f"""
            font-size: {self.font_config['metric']}px;
            font-weight: bold;
            color: {color};
        """)
        value_widget.setAlignment(Qt.AlignCenter)
        value_widget.setObjectName("value")
        
        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        widget.setLayout(layout)
        
        return widget
    
    def create_about_tab(self):
        """Create about/help tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        # Title
        title = QLabel("ℹ️ TENTANG APLIKASI")
        title.setStyleSheet(f"""
            font-size: {self.font_config['main_title']}px;
            font-weight: bold;
            color: #57cc99;
            padding: 15px;
            border-bottom: 3px solid #57cc99;
            background-color: #0f3460;
            border-radius: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Content
        content_frame = QFrame()
        content_frame.setMinimumHeight(550)
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 2px solid #57cc99;
                border-radius: 15px;
                padding: 30px;
                margin: 20px;
            }
        """)
        content_layout = QVBoxLayout()
        
        # App info
        info_html = f"""
        <div style='color: #e6e6e6; font-size: {self.font_config['tutorial_text']}px;'>
            <h2 style='color: #57cc99;'>Penerjemah Bahasa Isyarat Indonesia</h2>
            <p style='color: #80ed99;'>
                <strong>Inovasi Multimedia - Sistem Deteksi Real-time</strong>
            </p>
            
            <h3 style='color: #57cc99; margin-top: 20px;'>🎯 Fitur Utama:</h3>
            <ul style='color: #e6e6e6;'>
                <li>Deteksi real-time huruf A-Z menggunakan MediaPipe</li>
                <li>Klasifikasi dengan Neural Network (TensorFlow)</li>
                <li>Antarmuka multimedia yang interaktif</li>
                <li>Mode pembelajaran dengan panduan visual</li>
                <li>Analisis performa dan statistik deteksi</li>
                <li>Ekspor data dan screenshot</li>
            </ul>
            
            <h3 style='color: #57cc99; margin-top: 20px;'>🛠️ Teknologi:</h3>
            <ul style='color: #e6e6e6;'>
                <li><strong>Computer Vision:</strong> MediaPipe Hands, OpenCV</li>
                <li><strong>Machine Learning:</strong> TensorFlow, scikit-learn</li>
                <li><strong>GUI:</strong> PyQt5 dengan desain modern</li>
                <li><strong>Visualisasi:</strong> Matplotlib untuk analytics</li>
            </ul>
            
            <h3 style='color: #57cc99; margin-top: 20px;'>📖 Panduan Cepat:</h3>
            <ol style='color: #e6e6e6;'>
                <li>Pastikan kamera terhubung dan berfungsi</li>
                <li>Klik "Mulai Deteksi" untuk memulai</li>
                <li>Tunjukkan tangan dengan isyarat huruf A-Z</li>
                <li>Gunakan Mode Pembelajaran untuk belajar isyarat</li>
                <li>Pantau performa di tab Analisis</li>
            </ol>
            
            <div style='margin-top: 30px; padding: 15px; background-color: #0f3460; border-radius: 10px;'>
                <p style='color: #80ed99; text-align: center; border-radius; 10px'>
                    <strong>Mata Kuliah: Inovasi Multimedia</strong><br>
                    Dikembangkan untuk demonstrasi teknologi multimedia interaktif
                </p>
            </div>
        </div>
        """
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(info_html)
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                font-family: 'Arial', sans-serif;
            }
        """)
        
        content_layout.addWidget(info_text)
        content_frame.setLayout(content_layout)
        layout.addWidget(content_frame)
        
        # Developer info
        dev_frame = QFrame()
        dev_frame.setStyleSheet("""
            QFrame {
                background-color: #0f3460;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
            }
        """)

        tab.setLayout(layout)
        return tab

def main():
    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set High DPI scaling before creating any widgets
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set application icon and style untuk Windows
    app.setApplicationName("Penerjemah Bahasa Isyarat")
    app.setApplicationDisplayName("Inovasi Multimedia - Penerjemah Bahasa Isyarat")
    
    window = MultimediaSignLanguageApp()
    
    # Tampilkan maximize
    window.showMaximized()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()