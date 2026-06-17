# VisionEdge

A modern PyQt5 and OpenCV desktop application for image and video processing. Features interactive filters, contour detection methods, and real-time MP4 streaming with region zooming.

## Features
- Real-time video processing (MP4, AVI)
- Static image filtering (Gaussian, Mean, Median, Min, Max)
- Contour detection algorithms (Sobel, Prewitt, Roberts, Laplacian, Kirsch, Robinson, Canny)
- Interactive zoom overlay for high-resolution inspection
- Standalone executable build support

## Setup for Developers

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd image_processor_app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Building the Executable

To build the standalone `.exe` for end-users, use PyInstaller. Make sure you run this command from inside the `image_processor_app` directory:

```bash
pyinstaller --noconsole --onefile --icon=icon.png --add-data "icon.png;." --name=VisionEdge main.py
```
