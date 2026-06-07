# Safety Helmet and PPE Vest Detection System 👷‍♂️⚠️

A real-time Computer Vision Tkinter application built with YOLOv8 to monitor and enforce workplace safety by detecting helmets and PPE vests. Developed during a Computer Vision Internship at SAIL, Bhilai Steel Plant.

## ✨ Features

* **Multi-Mode:** Process live webcam feeds, static images, or pre-recorded videos.
* **Real-Time Alerts:** Triggers an automated audio beep when a person is detected without a helmet.
* **Auto-Save Logs:** Automatically captures and saves frames of safety violations (15-second cooldown to save storage).
* **Dual Models:** Easily switch between `light weight.pt` (high speed) and `most accurate.pt` (high precision).

## 📁 Repository Structure

* `helmet5.py`: Main application script (GUI + YOLO logic).
* `light weight.pt` & `most accurate.pt`: Custom trained YOLOv8 models.
* `requirements.txt`: Python dependencies.
* `user_manual.pdf`: System operation guide.

## ⚙️ Quick Setup

1. **Clone & Install Dependencies:**
```bash
git clone https://github.com/yourusername/safety-helmet-detection.git
cd safety-helmet-detection
pip install -r requirements.txt

```


2. **Update Local Paths:**
Open `helmet5.py` and update the hardcoded paths to match your system:
* **Model Path** (~Line 20): `self.model_path = r"path\to\light weight.pt"`
* **Save Directory** (~Line 33): `self.auto_save_folder = r"path\to\your\save\folder"`


3. **Run the Application:**
```bash
python helmet5.py

```



## 👨‍💻 Author

**Prince**
Computer Vision Project Intern @ Steel Authority of India Limited (SAIL), Bhilai Steel Plant (BSP)
