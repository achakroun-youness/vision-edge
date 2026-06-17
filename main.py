import sys
import cv2
import numpy as np
import qdarktheme
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, QFileDialog, QGroupBox, QDialog, QGraphicsView, QGraphicsScene, QSplashScreen)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import os
from PyQt5.QtCore import Qt, QTimer

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- Kernels Definition ---
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

kirsch_masks = [
    np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
    np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
    np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
    np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
    np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32)
]

robinson_masks = [
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32),
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
]


class ImageViewerDialog(QDialog):
    def __init__(self, qpixmap, title="Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 700)
        self.qpixmap = qpixmap
        
        layout = QVBoxLayout(self)
        
        controls_layout = QHBoxLayout()
        self.btn_save = QPushButton(" Save Image ")
        self.btn_save.setMinimumHeight(35)
        self.btn_save.setFont(QFont("Segoe UI", 10))
        self.btn_save.setStyleSheet("QPushButton { background-color: #28a745; color: white; border-radius: 4px; padding: 0px 15px; } QPushButton:hover { background-color: #218838; }")
        self.btn_save.clicked.connect(self.save_image)
        
        controls_layout.addWidget(self.btn_save)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        
        self.view.setScene(self.scene)
        self.scene.addPixmap(qpixmap)
        
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        layout.addWidget(self.view)
        
    def save_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp);;All Files (*)", options=options)
        if file_path:
            self.qpixmap.save(file_path)
            
    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.view.scale(zoom_factor, zoom_factor)

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionEdge")
        self.setGeometry(100, 100, 1300, 700)
        
        # Set Application Icon
        icon_path = resource_path("icon.png")
        self.setWindowIcon(QIcon(icon_path))
        
        self.media_path = None
        self.original_image = None
        self.filtered_image = None
        self.contour_image = None
        
        # Video properties
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_video_playing = False
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # --- Controls Box ---
        controls_group = QGroupBox("Processing Controls")
        controls_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(15, 20, 15, 15)
        controls_layout.setSpacing(15)
        
        self.btn_upload = QPushButton(" Upload Media (Img/Video) ")
        self.btn_upload.setMinimumHeight(40)
        self.btn_upload.setFont(QFont("Segoe UI", 10))
        self.btn_upload.clicked.connect(self.upload_media)
        controls_layout.addWidget(self.btn_upload)
        
        self.btn_play_pause = QPushButton(" Pause Video ")
        self.btn_play_pause.setMinimumHeight(40)
        self.btn_play_pause.setFont(QFont("Segoe UI", 10))
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)
        controls_layout.addWidget(self.btn_play_pause)
        
        lbl_filter = QLabel("Filter:")
        lbl_filter.setFont(QFont("Segoe UI", 10))
        controls_layout.addWidget(lbl_filter)
        
        self.combo_filter = QComboBox()
        self.combo_filter.setFont(QFont("Segoe UI", 10))
        self.combo_filter.setMinimumHeight(40)
        self.combo_filter.addItems(["None", "Grayscale", "Gaussian Blur", "Mean Blur", "Median Blur", "Min Filter (Erosion)", "Max Filter (Dilation)"])
        self.combo_filter.currentIndexChanged.connect(self.process_static_image_if_needed)
        controls_layout.addWidget(self.combo_filter)
        
        lbl_contour = QLabel("Contour Method:")
        lbl_contour.setFont(QFont("Segoe UI", 10))
        controls_layout.addWidget(lbl_contour)
        
        self.combo_contour = QComboBox()
        self.combo_contour.setFont(QFont("Segoe UI", 10))
        self.combo_contour.setMinimumHeight(40)
        self.combo_contour.addItems([
            "None", "Canny Edge", "Simple Thresholding", 
            "Simple Gradient", "Roberts Cross", "Prewitt", 
            "Sobel", "Laplacian", "Kirsch", "Robinson"
        ])
        self.combo_contour.currentIndexChanged.connect(self.process_static_image_if_needed)
        controls_layout.addWidget(self.combo_contour)
        

        
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        # --- Images Layout ---
        images_layout = QHBoxLayout()
        images_layout.setSpacing(20)
        
        # Original Image
        vbox_orig = QVBoxLayout()
        lbl_title_orig = QLabel("Original Media (Click to Zoom)")
        lbl_title_orig.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_title_orig.setAlignment(Qt.AlignCenter)
        vbox_orig.addWidget(lbl_title_orig)
        
        self.label_orig = QLabel()
        self.label_orig.setFixedSize(400, 400)
        self.label_orig.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e; border-radius: 8px;")
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setCursor(Qt.PointingHandCursor)
        self.label_orig.mousePressEvent = self.show_orig_image
        vbox_orig.addWidget(self.label_orig)
        images_layout.addLayout(vbox_orig)
        
        # Filtered Image
        vbox_filt = QVBoxLayout()
        lbl_title_filt = QLabel("Filtered Media (Click to Zoom)")
        lbl_title_filt.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_title_filt.setAlignment(Qt.AlignCenter)
        vbox_filt.addWidget(lbl_title_filt)
        
        self.label_filt = QLabel()
        self.label_filt.setFixedSize(400, 400)
        self.label_filt.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e; border-radius: 8px;")
        self.label_filt.setAlignment(Qt.AlignCenter)
        self.label_filt.setCursor(Qt.PointingHandCursor)
        self.label_filt.mousePressEvent = self.show_filt_image
        vbox_filt.addWidget(self.label_filt)
        images_layout.addLayout(vbox_filt)
        
        # Contour Image
        vbox_cont = QVBoxLayout()
        lbl_title_cont = QLabel("Detected Contours (Click to Zoom)")
        lbl_title_cont.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_title_cont.setAlignment(Qt.AlignCenter)
        vbox_cont.addWidget(lbl_title_cont)
        
        self.label_cont = QLabel()
        self.label_cont.setFixedSize(400, 400)
        self.label_cont.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e3e; border-radius: 8px;")
        self.label_cont.setAlignment(Qt.AlignCenter)
        self.label_cont.setCursor(Qt.PointingHandCursor)
        self.label_cont.mousePressEvent = self.show_cont_image
        vbox_cont.addWidget(self.label_cont)
        images_layout.addLayout(vbox_cont)
        
        main_layout.addLayout(images_layout)
        main_widget.setLayout(main_layout)

    def upload_media(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Media File", "", 
                                                   "Media Files (*.png *.jpg *.jpeg *.bmp *.tif *.mp4 *.avi)", options=options)
        if file_name:
            self.media_path = file_name
            
            # Stop existing video if any
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
            self.label_filt.clear()
            self.label_cont.clear()
            self.filtered_image = None
            self.contour_image = None
            
            if file_name.lower().endswith(('.mp4', '.avi')):
                # It's a video
                self.cap = cv2.VideoCapture(self.media_path)
                self.is_video_playing = True
                self.btn_play_pause.setEnabled(True)
                self.btn_play_pause.setText(" Pause Video ")
                self.timer.start(33) # ~30fps
            else:
                # It's an image
                self.btn_play_pause.setEnabled(False)
                self.is_video_playing = False
                self.original_image = cv2.imread(self.media_path, cv2.IMREAD_COLOR)
                if self.original_image is not None:
                    self.display_image(self.original_image, self.label_orig)
                    self.process_image()

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame
                self.display_image(self.original_image, self.label_orig)
                self.process_image()
            else:
                # Loop video when it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def toggle_play_pause(self):
        if self.cap is not None:
            if self.is_video_playing:
                self.timer.stop()
                self.is_video_playing = False
                self.btn_play_pause.setText(" Play Video ")
            else:
                self.timer.start(33)
                self.is_video_playing = True
                self.btn_play_pause.setText(" Pause Video ")

    def process_static_image_if_needed(self):
        if not self.is_video_playing and self.original_image is not None:
            self.process_image()

    def process_image(self):
        if self.original_image is None:
            return
            
        # --- 1. Filter Logic ---
        filter_type = self.combo_filter.currentText()
        img_for_filter = self.original_image.copy()
        
        if filter_type == "None":
            self.filtered_image = img_for_filter
        elif filter_type == "Grayscale":
            gray_img = cv2.cvtColor(img_for_filter, cv2.COLOR_BGR2GRAY)
            self.filtered_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        elif filter_type == "Gaussian Blur":
            self.filtered_image = cv2.GaussianBlur(img_for_filter, (15, 15), 0)
        elif filter_type == "Mean Blur":
            self.filtered_image = cv2.blur(img_for_filter, (15, 15))
        elif filter_type == "Median Blur":
            self.filtered_image = cv2.medianBlur(img_for_filter, 15)
        elif filter_type == "Min Filter (Erosion)":
            kernel = np.ones((5, 5), np.uint8)
            self.filtered_image = cv2.erode(img_for_filter, kernel, iterations=1)
        elif filter_type == "Max Filter (Dilation)":
            kernel = np.ones((5, 5), np.uint8)
            self.filtered_image = cv2.dilate(img_for_filter, kernel, iterations=1)
            
        self.display_image(self.filtered_image, self.label_filt)
        
        # --- 2. Contour Detection Logic ---
        contour_type = self.combo_contour.currentText()
        if contour_type == "None":
            self.label_cont.clear()
            self.contour_image = None
            return
            
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        edges = None
        
        if contour_type == "Canny Edge":
            edges = cv2.Canny(gray, 100, 200)
        elif contour_type == "Simple Thresholding":
            _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            magnitude = None
            if contour_type == "Simple Gradient":
                kx = np.array([[-1, 1]])
                ky = np.array([[-1], [1]])
                gx = cv2.filter2D(gray, cv2.CV_64F, kx)
                gy = cv2.filter2D(gray, cv2.CV_64F, ky)
                magnitude = cv2.magnitude(gx, gy)
            elif contour_type == "Roberts Cross":
                gx = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
                gy = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
                magnitude = cv2.magnitude(gx, gy)
            elif contour_type == "Prewitt":
                gx = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
                gy = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
                magnitude = cv2.magnitude(gx, gy)
            elif contour_type == "Sobel":
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(gx, gy)
            elif contour_type == "Laplacian":
                magnitude = cv2.Laplacian(gray, cv2.CV_64F)
                magnitude = np.abs(magnitude)
            elif contour_type == "Kirsch":
                responses = [cv2.filter2D(gray, cv2.CV_64F, mask) for mask in kirsch_masks]
                magnitude = np.max(responses, axis=0)
            elif contour_type == "Robinson":
                responses = [cv2.filter2D(gray, cv2.CV_64F, mask) for mask in robinson_masks]
                magnitude = np.max(responses, axis=0)
            
            if magnitude is not None:
                magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                magnitude_8u = np.uint8(magnitude)
                _, edges = cv2.threshold(magnitude_8u, 50, 255, cv2.THRESH_BINARY)
                
        if edges is not None:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered_contours = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= 15 or h >= 15: 
                    filtered_contours.append(cnt)
                    
            self.contour_image = self.original_image.copy()
            cv2.drawContours(self.contour_image, filtered_contours, -1, (0, 255, 0), 2)
            
            self.display_image(self.contour_image, self.label_cont)

    def display_image(self, img, label):
        if img is None:
            return
        
        h, w = img.shape[:2]
        max_dim = 398 
        
        if w == 0 or h == 0: return
            
        scale = min(max_dim/w, max_dim/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w == 0 or new_h == 0: return
            
        resized_img = cv2.resize(img, (new_w, new_h))
        
        if len(resized_img.shape) == 3:
            rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * new_w
            q_img = QImage(rgb_image.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = new_w
            q_img = QImage(resized_img.data, new_w, new_h, bytes_per_line, QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)

    # --- Click Actions for Zooming ---
    def show_orig_image(self, event):
        if self.original_image is not None:
            self.show_image_dialog(self.original_image, "Original Image Viewer")
            
    def show_filt_image(self, event):
        if self.filtered_image is not None:
            self.show_image_dialog(self.filtered_image, "Filtered Image Viewer")
            
    def show_cont_image(self, event):
        if self.contour_image is not None:
            self.show_image_dialog(self.contour_image, "Detected Contours Viewer")
            
    def show_image_dialog(self, cv_img, title):
        h, w = cv_img.shape[:2]
        if len(cv_img.shape) == 3:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(q_img)
        dialog = ImageViewerDialog(pixmap, title, self)
        
        # Pause video when dialog opens, then resume
        was_playing = self.is_video_playing
        if was_playing: self.toggle_play_pause()
        
        dialog.exec_()
        
        if was_playing: self.toggle_play_pause()

if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    
    # --- Splash Screen ---
    icon_path = resource_path("icon.png")
    splash_pix = QPixmap(icon_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    
    splash.showMessage("Loading VisionEdge...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    app.processEvents()
    
    # Fake loading delay to display the splash animation
    time.sleep(2) 
    
    window = ImageProcessorApp()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())
