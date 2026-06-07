```markdown
# Safety Helmet and PPE Vest Detection System 👷‍♂️⚠️

A real-time Computer Vision Tkinter application built with YOLOv8 to monitor and enforce workplace safety by detecting helmets and PPE vests. Developed during a Computer Vision Internship at the Steel Authority of India Limited (SAIL), Bhilai Steel Plant (BSP).

## ✨ Features
* **Multi-Mode Detection:** Process live webcam feeds, static images, or pre-recorded videos.
* **Real-Time Alerts:** Triggers an automated audio beep when a person is detected without a helmet.
* **Auto-Save Logs:** Automatically captures and saves frames of safety violations (operates on a 15-second cooldown to save storage).
* **Dual Models:** Easily switch between high-speed inference and maximum precision.

## 📥 Download Model Weights
Due to GitHub's file size limits, the trained YOLOv8 `.pt` models are hosted externally. **You must download these and place them in the same folder as `helmet5.py` before running.**

* ⚡ **[Download `light weight.pt`](https://icedrive.net/s/Vak1V7jFSgDB67u5t4hXtkXZYC34)** - Optimized for faster inference and higher FPS.
* 🎯 **[Download `most accurate.pt`](https://icedrive.net/s/3j6f51RQVZ391D6vANhP5T23wi1a)** - Heavy model optimized for maximum detection accuracy.

## ⚙️ Quick Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/safety-helmet-detection.git](https://github.com/yourusername/safety-helmet-detection.git)
   cd safety-helmet-detection

```

2. **Download Models:**
Download the two `.pt` files from the links above and move them into this project folder.
3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Update Local Paths:**
Open `helmet5.py` in any text editor and update the hardcoded paths to match your system:
* **Model Path** (~Line 20): `self.model_path = r"path\to\your\downloaded\light weight.pt"`
* **Save Directory** (~Line 33): `self.auto_save_folder = r"path\to\your\desired\save\folder"`


5. **Run the Application:**
```bash
python helmet5.py

```



## 👨‍💻 Author

**Prince**
Computer Vision Project Intern @ Steel Authority of India Limited (SAIL), Bhilai Steel Plant (BSP)

```

```
