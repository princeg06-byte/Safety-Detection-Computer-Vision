import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import numpy as np
import os
import time
import winsound
import queue


class SafetyHelmetAndPPEVestDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Safety Helmet and PPE Vest Detection System")
        self.root.geometry("900x700")

        # Model variables
        self.model = None

        self.model_path = r"C:\Users\princ\OneDrive\Desktop\light weight.pt"
        # self.model_path = r"path\to\your\model.pt"
        # above path is for best file
        # above path will be different for different system

        # Camera variables
        self.cap = None
        self.is_camera_running = False
        self.webcam_window = None
        self.current_frame = None

        # Alarm and display variables
        self.last_beep_time = 0  # Track last alarm time
        self.frame_queue = queue.Queue(maxsize=1)  # Queue for thread-safe frame updates

        # Auto-save variables
        self.last_auto_save_time = 0  # Track last auto-save time

        self.auto_save_folder = r"C:\Users\princ\OneDrive\Desktop\project screenshots\auto-save no-helmet images"
        # self.auto_save_folder = r"path\to\your\save\folder"
        # above path is for where to save the no-helmet pics
        # above path will be different for different system

        self.ensure_auto_save_folder_exists()

        # Show mode selection dialog first
        self.show_mode_selection()

        # Create GUI elements
        self.create_widgets()

        # Load model on startup
        self.load_model()

        # Start periodic display update
        self.root.after(33, self.update_display)

    def ensure_auto_save_folder_exists(self):
        """Create auto-save folder if it doesn't exist"""
        try:
            if not os.path.exists(self.auto_save_folder):
                os.makedirs(self.auto_save_folder)
                self.log_result(f"Created auto-save folder: {self.auto_save_folder}")
        except Exception as e:
            self.log_result(f"Error creating auto-save folder: {str(e)}")

    def auto_save_no_helmet_image(self, frame):
        """Auto-save image when no helmet detected (15-second restriction)"""
        try:
            current_time = time.time()
            # Check if 15 seconds have passed since last auto-save
            if current_time - self.last_auto_save_time >= 15:
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                filename = f"no_helmet_detected_{timestamp}.jpg"
                filepath = os.path.join(self.auto_save_folder, filename)

                # Save the image
                cv2.imwrite(filepath, frame)
                self.last_auto_save_time = current_time

                self.thread_safe_log(f"Auto-saved: {filename}")
                return True
        except Exception as e:
            self.thread_safe_log(f"Auto-save error: {str(e)}")
        return False

    def show_mode_selection(self):
        """Show mode selection dialog at startup"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Detection Mode")
        dialog.geometry("400x300")
        dialog.grab_set()

        dialog.transient(self.root)
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")

        title_label = ttk.Label(dialog, text="Safety Helmet and PPE Vest Detection System",
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        subtitle_label = ttk.Label(dialog, text="Choose your detection mode:",
                                   font=("Arial", 12))
        subtitle_label.pack(pady=10)

        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=20)

        webcam_btn = ttk.Button(buttons_frame, text="📷 Live Webcam Detection",
                                command=lambda: self.select_mode(dialog, 'webcam'),
                                width=25)
        webcam_btn.pack(pady=10)

        image_btn = ttk.Button(buttons_frame, text="🖼️ Detect in Images",
                               command=lambda: self.select_mode(dialog, 'image'),
                               width=25)
        image_btn.pack(pady=10)

        video_btn = ttk.Button(buttons_frame, text="🎥 Detect in Videos",
                               command=lambda: self.select_mode(dialog, 'video'),
                               width=25)
        video_btn.pack(pady=10)

        all_btn = ttk.Button(buttons_frame, text="⚙️ All Detection Modes",
                             command=lambda: self.select_mode(dialog, 'all'),
                             width=25)
        all_btn.pack(pady=10)

        instructions = ttk.Label(dialog,
                                 text="Select your preferred mode or choose 'All Detection Modes'\nfor full functionality",
                                 font=("Arial", 9), justify=tk.CENTER)
        instructions.pack(pady=20)

        self.selected_mode = None
        dialog.wait_window()

    def select_mode(self, dialog, mode):
        """Handle mode selection"""
        self.selected_mode = mode
        dialog.destroy()

        if mode == 'webcam':
            self.root.after(1000, self.auto_start_webcam)
        elif mode == 'image':
            self.root.after(1000, self.detect_image)
        elif mode == 'video':
            self.root.after(1000, self.detect_video)

    def auto_start_webcam(self):
        """Auto start webcam for webcam-only mode"""
        if self.model and self.selected_mode == 'webcam':
            self.toggle_webcam()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        mode_text = ""
        if hasattr(self, 'selected_mode'):
            if self.selected_mode == 'webcam':
                mode_text = " - Live Webcam Mode"
            elif self.selected_mode == 'image':
                mode_text = " - Image Detection Mode"
            elif self.selected_mode == 'video':
                mode_text = " - Video Detection Mode"
            elif self.selected_mode == 'all':
                mode_text = " - All Modes Available"

        title_label = ttk.Label(main_frame, text=f"Safety Helmet and PPE Vest Detection System{mode_text}",
                                font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="5")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value=self.model_path)
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)
        ttk.Button(model_frame, text="Reload Model", command=self.load_model).grid(row=0, column=3, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(model_frame, textvariable=self.status_var, foreground="green")
        status_label.grid(row=1, column=0, columnspan=4, pady=5)

        conf_frame = ttk.Frame(main_frame)
        conf_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(conf_frame, text="Confidence Threshold:").grid(row=0, column=0)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.conf_var,
                               orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, padx=10)
        self.conf_label = ttk.Label(conf_frame, text="0.5")
        self.conf_label.grid(row=0, column=2)
        conf_scale.configure(command=self.update_conf_label)

        if not hasattr(self, 'selected_mode') or self.selected_mode == 'all':
            modes_frame = ttk.LabelFrame(main_frame, text="Detection Modes", padding="10")
            modes_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            self.webcam_btn = ttk.Button(modes_frame, text="Start Webcam",
                                         command=self.toggle_webcam, width=15)
            self.webcam_btn.grid(row=0, column=0, padx=5)

            ttk.Button(modes_frame, text="Detect in Image",
                       command=self.detect_image, width=15).grid(row=0, column=1, padx=5)

            ttk.Button(modes_frame, text="Detect in Video",
                       command=self.detect_video, width=15).grid(row=0, column=2, padx=5)
        else:
            single_mode_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            single_mode_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            if self.selected_mode == 'webcam':
                self.webcam_btn = ttk.Button(single_mode_frame, text="Start Webcam",
                                             command=self.toggle_webcam, width=20)
                self.webcam_btn.grid(row=0, column=0, padx=5)

                ttk.Button(single_mode_frame, text="Switch to All Modes",
                           command=self.switch_to_all_modes, width=20).grid(row=0, column=1, padx=5)
            elif self.selected_mode == 'image':
                ttk.Button(single_mode_frame, text="Select Image",
                           command=self.detect_image, width=20).grid(row=0, column=0, padx=5)

                ttk.Button(single_mode_frame, text="Switch to All Modes",
                           command=self.switch_to_all_modes, width=20).grid(row=0, column=1, padx=5)
            elif self.selected_mode == 'video':
                ttk.Button(single_mode_frame, text="Select Video",
                           command=self.detect_video, width=20).grid(row=0, column=0, padx=5)

                ttk.Button(single_mode_frame, text="Switch to All Modes",
                           command=self.switch_to_all_modes, width=20).grid(row=0, column=1, padx=5)

        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="5")
        video_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.video_label = ttk.Label(video_frame, text="Video feed will appear here",
                                     anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0)

        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="5")
        results_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.results_text = tk.Text(results_frame, height=8, width=70)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        main_frame.rowconfigure(5, weight=1)

    def switch_to_all_modes(self):
        """Switch from single mode to all modes interface"""
        self.selected_mode = 'all'
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_widgets()
        if self.model:
            self.status_var.set("Model loaded successfully!")
            self.log_result("Switched to all modes interface")
            self.log_result(f"Classes: {self.model.names}")

    def update_conf_label(self, value):
        self.conf_label.config(text=f"{float(value):.2f}")

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select YOLOv8 Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)

    def load_model(self):
        try:
            model_path = self.model_path_var.get()
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return

            self.status_var.set("Loading model...")
            self.root.update()

            self.model = YOLO(model_path)
            self.status_var.set("Model loaded successfully!")
            self.log_result(f"Model loaded: {model_path}")
            self.log_result(f"Classes: {self.model.names}")

        except Exception as e:
            self.status_var.set("Error loading model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def toggle_webcam(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        if not self.is_camera_running:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            self.is_camera_running = True
            self.webcam_btn.config(text="Stop Webcam")
            self.log_result("Webcam started")

            self.create_webcam_window()

            self.webcam_thread = threading.Thread(target=self.webcam_loop)
            self.webcam_thread.daemon = True
            self.webcam_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")

    def create_webcam_window(self):
        """Create a separate window for webcam display"""
        if self.webcam_window is not None:
            try:
                self.webcam_window.destroy()
            except:
                pass

        self.webcam_window = tk.Toplevel(self.root)
        self.webcam_window.title("Safety Helmet and PPE Vest Detection - Live Webcam")
        self.webcam_window.geometry("800x600")

        self.webcam_display = ttk.Label(self.webcam_window, text="Starting webcam...", anchor=tk.CENTER)
        self.webcam_display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        instructions = ttk.Label(self.webcam_window,
                                 text="Press 'S' to save current frame | Press 'Q' to quit webcam | Auto-save: ON (15s interval)",
                                 font=("Arial", 10))
        instructions.pack(pady=5)

        self.webcam_window.protocol("WM_DELETE_WINDOW", self.on_webcam_window_close)

        self.webcam_window.bind('<KeyPress-q>', lambda e: self.stop_webcam())
        self.webcam_window.bind('<KeyPress-Q>', lambda e: self.stop_webcam())
        self.webcam_window.bind('<KeyPress-s>', lambda e: self.save_current_frame())
        self.webcam_window.bind('<KeyPress-S>', lambda e: self.save_current_frame())
        self.webcam_window.focus_set()

    def on_webcam_window_close(self):
        """Handle webcam window close event"""
        self.stop_webcam()

    def save_current_frame(self):
        """Save current webcam frame"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            filename = f"webcam_detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.log_result(f"Frame saved as {filename}")
            messagebox.showinfo("Saved", f"Frame saved as {filename}")

    def stop_webcam(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'webcam_btn'):
            self.webcam_btn.config(text="Start Webcam")
        if self.webcam_window:
            self.webcam_window.destroy()
            self.webcam_window = None
        self.video_label.config(image="", text="Video feed will appear here")
        self.log_result("Webcam stopped")

    def webcam_loop(self):
        frame_count = 0
        while self.is_camera_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.current_frame = frame.copy()

                results = self.model(frame, conf=self.conf_var.get())

                annotated_frame = self.draw_detections(frame, results[0])

                # Count persons and helmets
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    person_count = sum(1 for c in classes if self.model.names[c] == "person")
                    helmet_count = sum(1 for c in classes if self.model.names[c] == "helmet")
                else:
                    person_count = 0
                    helmet_count = 0

                # Trigger alarm and auto-save if person detected without helmet
                if person_count > helmet_count:
                    current_time = time.time()

                    # Trigger alarm with 2-second cooldown
                    if current_time - self.last_beep_time > 2:
                        winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
                        self.last_beep_time = current_time
                        self.thread_safe_log("Alarm: Person without helmet detected!")

                    # Auto-save image with 15-second restriction
                    self.auto_save_no_helmet_image(frame)

                cv2.putText(annotated_frame, f"Frame: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, "Press 'S' to save, 'Q' to quit",
                            (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.current_frame = annotated_frame.copy()

                # Put frame into queue for display
                try:
                    self.frame_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass

                frame_count += 1

            except Exception as e:
                self.log_result(f"Webcam error: {str(e)}")
                break

    def update_display(self):
        """Update GUI display with frames from the queue"""
        try:
            annotated_frame = self.frame_queue.get_nowait()

            # For webcam window
            if self.webcam_window and self.webcam_window.winfo_exists():
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                window_width = 760
                window_height = 500
                pil_image = pil_image.resize((window_width, window_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                self.webcam_display.config(image=photo)
                self.webcam_display.image = photo

            # For main window
            rgb_frame_small = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image_small = Image.fromarray(rgb_frame_small)
            pil_image_small = pil_image_small.resize((300, 200), Image.Resampling.LANCZOS)
            photo_small = ImageTk.PhotoImage(pil_image_small)
            self.video_label.config(image=photo_small)
            self.video_label.image = photo_small

        except queue.Empty:
            pass

        self.root.after(33, self.update_display)  # ~30 fps

    def thread_safe_log(self, message):
        """Log messages in a thread-safe manner"""
        self.root.after(0, lambda: self.log_result(message))

    def detect_image(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if filename:
            try:
                image = cv2.imread(filename)
                results = self.model(image, conf=self.conf_var.get())
                annotated_image = self.draw_detections(image, results[0])
                self.display_result_image(annotated_image)
                self.log_detection_results(results[0], f"Image: {os.path.basename(filename)}")

                if messagebox.askyesno("Save Result", "Do you want to save the detection result?"):
                    save_path = filedialog.asksaveasfilename(
                        defaultextension=".jpg",
                        filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
                    )
                    if save_path:
                        cv2.imwrite(save_path, annotated_image)
                        self.log_result(f"Result saved: {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def detect_video(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if filename:
            output_path = filedialog.asksaveasfilename(
                title="Save Detection Result As",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
            )

            if output_path:
                thread = threading.Thread(target=self.process_video, args=(filename, output_path))
                thread.daemon = True
                thread.start()

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            self.log_result(f"Processing video: {total_frames} frames")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.conf_var.get())
                annotated_frame = self.draw_detections(frame, results[0])
                out.write(annotated_frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.log_result(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

            cap.release()
            out.release()

            self.log_result(f"Video processing completed: {output_path}")
            messagebox.showinfo("Success", "Video processing completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {str(e)}")

    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        annotated_image = image.copy()

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names.get(cls, f"Class_{cls}")
                color = colors[cls % len(colors)]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated_image

    def display_result_image(self, image):
        """Display detection result in a new window"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        max_size = 800
        if pil_image.width > max_size or pil_image.height > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        result_window = tk.Toplevel(self.root)
        result_window.title("Detection Result")

        photo = ImageTk.PhotoImage(pil_image)
        label = ttk.Label(result_window, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)

    def log_detection_results(self, results, source):
        """Log detection results to text area"""
        self.log_result(f"\n--- Detection Results for {source} ---")

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            self.log_result(f"Total detections: {len(classes)}")

            class_counts = {}
            for cls, conf in zip(classes, confidences):
                class_name = self.model.names.get(cls, f"Class_{cls}")
                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(conf)

            for class_name, confs in class_counts.items():
                avg_conf = np.mean(confs)
                self.log_result(f"- {class_name}: {len(confs)} detections (avg conf: {avg_conf:.3f})")
        else:
            self.log_result("No detections found")

    def log_result(self, message):
        """Add a message to the results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.results_text.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = SafetyHelmetAndPPEVestDetectionGUI(root)
    root.mainloop()
