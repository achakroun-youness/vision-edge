# VisionEdge 🔵

> An academic computer vision desktop application for real-time image & video filtering and contour detection.

Built with **Python**, **PyQt5**, and **OpenCV**.

---

## 👥 Team

| Name |
|------|
| **Achakroun Youness** |
| **Bidnaben Wafa** |
| **Alla Maryem** |
| **Nouzha Ait Hsina** |
| **Boudaoud Amine** |

**Course:** MIAE S2 — Traitement d'images  
**Repository:** [https://github.com/achakroun-youness/vision-edge](https://github.com/achakroun-youness/vision-edge)

---

## ✨ Features

- 📁 Upload **images** (PNG, JPG, BMP, TIF) or **videos** (MP4, AVI)
- 🎛️ **5 image filters**: Gaussian, Mean, Median, Erosion (Min), Dilation (Max)
- 🔍 **9 contour detection methods**: Canny, Thresholding, Gradient, Roberts Cross, Prewitt, Sobel, Laplacian, Kirsch, Robinson
- 🚫 Automatically filters out small contours **(< 15px)**
- 🔎 Click any image panel to open it in a **zoomable overlay**
- 💾 Save any processed image to disk
- 🎥 Real-time video frame processing with **Pause/Play** control
- 🌑 Dark mode UI with animated **splash screen**

---

## 🚀 Getting Started (for Developers)

### Prerequisites
- Python **3.10+** installed on your machine
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/achakroun-youness/vision-edge.git
cd vision-edge
```

### 2. (Recommended) Create a Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

---

## 📦 Building a Standalone Executable (Windows)

To share the app with users who do not have Python installed, use PyInstaller:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --icon=icon.png --add-data "icon.png;." --name=VisionEdge main.py
```

The final `VisionEdge.exe` will be inside the `dist/` folder.

---

## 🗂️ Project Structure

```
vision-edge/
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── icon.png             # Application icon
├── .gitignore           # Git ignore file
└── README.md            # This file
```

---

## 📚 Academic Context

This project is part of the **MIAE S2 - Image Processing** course. 
It demonstrates classical computer vision techniques for filtering and edge/contour detection.
