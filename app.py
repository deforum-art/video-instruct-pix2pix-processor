import datetime
import re
import secrets
import sys
import traceback

import cv2
import os

import numpy as np
import torch
from PySide6.QtCore import Qt, Signal, QSize, Slot, QRunnable, QObject, QThreadPool
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, \
    QWidget, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QGridLayout, QListWidgetItem, QListWidget
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionUpscalePipeline

class MainWindow(QMainWindow):
    output_signal = Signal(object)
    grid_signal = Signal(object)
    def __init__(self):
        super().__init__()

        # Initialize the UI
        self.setWindowTitle("Video Chopper")
        self.setFixedSize(400, 300)

        # Initialize the video file path and grid file path
        self.video_path = None
        self.grid_path = None

        # Create the load video button
        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)

        # Create the load folder button
        self.load_folder_button = QPushButton("Load Folder", self)
        self.load_folder_button.setEnabled(True)
        self.load_folder_button.clicked.connect(self.load_folder)

        # Create the video preview label
        self.video_preview_label = QLabel("No video loaded", self)
        self.video_preview_label.setAlignment(Qt.AlignCenter)
        self.video_preview_label.setFixedSize(320, 240)

        # Create the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_preview_label)

        # Create the button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_video_button)
        button_layout.addWidget(self.load_folder_button)
        main_layout.addLayout(button_layout)
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        self.upscaler = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler").to("cuda")
        # Create the central widget and set the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.pipewidget = PipelineWidget()
        self.pipewidget.show()
        self.grid_signal.connect(self.grid_preview_func)
        self.output_signal.connect(self.output_preview_func)
        self.threadpool = QThreadPool()
        self.init_process_window()
    def init_process_window(self):
        self.process_window = QWidget()
        layout = QGridLayout()
        self.process_window.setLayout(layout)
        self.input_list = QListWidget()
        self.input_list.setIconSize(QSize(256, 256))
        self.output_list = QListWidget()
        self.output_list.setIconSize(QSize(256, 256))
        layout.addWidget(QLabel("Input Images"), 0, 0)
        layout.addWidget(self.input_list, 1, 0)
        layout.addWidget(QLabel("Cropped Images"), 0, 1)
        layout.addWidget(self.output_list, 1, 1)
        self.process_window.show()

    def init_preview_window(self):
        self.window = QWidget()
        layout = QHBoxLayout()

        list_widget = QListWidget()
        layout.addWidget(list_widget)
        list_widget.setIconSize(QSize(256, 256))
        self.window.setLayout(layout)
        self.window.show()
        return list_widget

    @Slot(object)
    def grid_preview_func(self, image):
        self.input_list.addItem(image)
    @Slot(object)
    def output_preview_func(self, image):
        self.output_list.addItem(image)

    def load_video(self):
        # Open a file dialog to select a video file
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi)")

        if not file_path:
            # The user cancelled the file dialog
            return

        # Open the video file using cv2
        cap = cv2.VideoCapture(file_path)

        # Define the target frame size and grid size
        target_size = (self.pipewidget.image_width.value(), self.pipewidget.image_height.value())
        grid_size = (self.pipewidget.grid_width.value(), self.pipewidget.grid_height.value())

        # Initialize a buffer for the grid
        grid = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)

        # Initialize the current grid position
        grid_pos = (0, 0)

        # Create a new window with a QListWidget
        list_widget = self.init_preview_window()
        # Loop over frames in the video
        frame_num = 0
        os.makedirs("grid", exist_ok=True)
        while True:
            # Read the next frame from the video
            ret, frame = cap.read()

            if not ret:
                # End of the video
                break

            # Resize the frame to the target size
            frame = cv2.resize(frame, target_size)

            # Add the frame to the grid
            x, y = grid_pos
            grid[y:y + target_size[1], x:x + target_size[0], :] = frame

            # Update the grid position
            grid_pos = (x + target_size[0], y)
            if grid_pos[0] >= grid_size[0]:
                grid_pos = (0, y + target_size[1])

            # If the grid is full, save it as a PNG file and add it to the QListWidget
            if grid_pos[1] >= grid_size[1]:
                grid_filename = f"grid/grid_{frame_num:04}.png"
                cv2.imwrite(grid_filename, grid)

                # Reset the grid buffer and position
                grid = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
                grid_pos = (0, 0)

                # Create a new QListWidgetItem with the grid as an icon
                pixmap = QPixmap(grid_filename).scaled(256, 256)
                item = QListWidgetItem(QIcon(pixmap), "", list_widget)
                item.setToolTip(grid_filename)

                # Increment the frame counter
                frame_num += 1

        # If there are leftover frames in the buffer, save them as a PNG file and add them to the QListWidget
        if grid_pos != (0, 0):
            grid_filename = f"grid_{frame_num:04}.png"
            cv2.imwrite(grid_filename, grid)

            # Create a new QListWidgetItem with the grid as an icon
            pixmap = QPixmap(grid_filename).scaled(256, 256)
            item = QListWidgetItem(QIcon(pixmap), "", list_widget)
            item.setToolTip(grid_filename)

        # Release the video file
        cap.release()

    def load_folder(self):
        if not self.grid_path:
            # Open a file dialog to select a folder
            folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")

            if not folder_path:
                # The user cancelled the file dialog
                return

            # Update the grid path
            self.grid_path = folder_path


        # Assemble the grid PNG files into an animated GIF
        worker = Worker(self.assemble_gif_from_grids)
        self.threadpool.start(worker)
        #self.assemble_gif_from_grids()

    def preview_video_frame(self):
        # Open the video file using cv2
        cap = cv2.VideoCapture(self.video_path)

        # Read the first frame from the video
        ret, frame = cap.read()

        if not ret:
            # There was an error reading the first frame
            return

        # Resize the frame to fit the preview label
        frame = cv2.resize(frame, (320, 240))

        # Convert the frame to a QPixmap and display it in the preview label
        qimage = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.video_preview_label.setPixmap(pixmap)

        # Release the video file
        cap.release()

    def chop_video_to_frames(self, video_path, grid_path, target_size=(128, 128), grid_size=(1024, 1024)):
        # Open the video file using cv2
        cap = cv2.VideoCapture(video_path)

        # Initialize the buffer for the grid
        grid = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)

        # Initialize the current grid position and frame counter
        grid_pos = (0, 0)
        frame_num = 1
        os.makedirs("grid", exist_ok=True)

        # Loop over frames in the video
        while True:
            # Read the next frame from the video
            ret, frame = cap.read()

            if not ret:
                # End of the video
                break

            # Resize the frame to the target size
            frame = cv2.resize(frame, target_size)

            # Add the frame to the grid
            x, y = grid_pos
            grid[y:y + target_size[1], x:x + target_size[0], :] = frame

            # Update the grid position
            grid_pos = (x + target_size[0], y)
            if grid_pos[0] >= grid_size[0]:
                grid_pos = (0, y + target_size[1])

            # If the grid is full, save it as a PNG file
            if grid_pos[1] >= grid_size[1]:
                grid_filename = os.path.join(grid_path, f"grid/grid_{frame_num:04}.png")
                cv2.imwrite(grid_filename, grid)

                # Reset the grid buffer and position
                grid = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
                grid_pos = (0, 0)

                # Increment the frame counter
                frame_num += 1

        # If there are leftover frames in the buffer, save them as a PNG file
        if grid_pos != (0, 0):
            grid_filename = os.path.join(grid_path, f"grid_{frame_num:04}.png")
            cv2.imwrite(grid_filename, grid)

        # Release the video file
        cap.release()

    def assemble_gif_from_grids(self, progress_callback=None):
        # Get a list of all the grid PNG files in the folder
        grid_files = [filename for filename in os.listdir(self.grid_path) if filename.endswith("png")]

        if not grid_files:
            # There are no grid files in the folder
            return

        # Sort the grid files by frame number
        grid_files.sort(key=lambda filename: int(re.findall(r'\d+', filename)[-1]))

        # Define the target frame size and grid size
        target_size = (self.pipewidget.image_width.value(), self.pipewidget.image_height.value())
        grid_size = (1024, 1024)
        # Initialize the list of frames
        frames = []
        try:
            seed = int(self.pipewidget.generator_edit.text())
        except:
            seed = secrets.randbelow(4444444)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        # Loop over the grid files and add each one to the list of frames
        for grid_filename in grid_files:
            grid_path = os.path.join(self.grid_path, grid_filename)
            grid_image = Image.open(grid_path)

            grid_image = self.pipe(prompt=self.pipewidget.prompt_edit.text(),
                                   image=grid_image,
                                   num_inference_steps=self.pipewidget.num_inference_steps_spin.value(),
                                   guidance_scale=self.pipewidget.guidance_scale_spin.value(),
                                   image_guidance_scale=self.pipewidget.image_guidance_scale_spin.value(),
                                   generator=generator
                                   ).images[0]

            input_image_np = np.array(grid_image)
            input_image_pixmap = QPixmap.fromImage(
                QImage(input_image_np.data, input_image_np.shape[1], input_image_np.shape[0], QImage.Format_RGB888))
            input_item = QListWidgetItem(QIcon(input_image_pixmap), "")
            self.grid_signal.emit(input_item)
            #input_list.addItem(input_item)

            for y in range(0, grid_image.height, target_size[1]):
                for x in range(0, grid_image.width, target_size[0]):
                    frame = grid_image.crop((x, y, x+target_size[0], y+target_size[1]))
                    input_image_np = np.array(frame)
                    input_image_pixmap = QPixmap.fromImage(
                        QImage(input_image_np.data, input_image_np.shape[1], input_image_np.shape[0],
                               QImage.Format_RGB888))
                    input_item = QListWidgetItem(QIcon(input_image_pixmap), "")
                    self.output_signal.emit(input_item)
                    #output_list.addItem(input_item)
                    """frame = self.upscaler(prompt= self.pipewidget.negative_prompt_edit.text(),
                                            image = frame,
                                            #num_inference_steps = self.pipewidget.num_inference_steps_spin.value(),
                                            guidance_scale= self.pipewidget.guidance_scale_spin.value(),
                                            noise_level = 20,
                                            negative_prompt = None,
                                            num_images_per_prompt = 1,
                                            eta=0.0,
                                            generator=generator).images[0]"""

                    frames.append(frame)
        # Calculate the frame duration based on the number of frames
        frame_duration = int(1000 / (len(frames) / 10))  # 10 frames per second
        # Save the frames as an animated GIF
        os.makedirs("animations", exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gif_filename = os.path.join("animations", f"animation_{time_str}.gif")
        frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)


class PipelineWidget(QWidget):
    parameters_updated = Signal(dict)

    def __init__(self):
        super().__init__()

        # Define the available options for the output_type parameter
        self.output_type_options = ["pil", "np.array"]

        # Create the widgets to control each parameter
        self.prompt_edit = QLineEdit()
        self.image_edit = QLineEdit()
        self.num_inference_steps_spin = QSpinBox()
        self.guidance_scale_spin = QDoubleSpinBox()
        self.image_guidance_scale_spin = QDoubleSpinBox()
        self.negative_prompt_edit = QLineEdit()
        self.num_images_per_prompt_spin = QSpinBox()
        self.eta_spin = QDoubleSpinBox()
        self.generator_edit = QLineEdit()
        self.latents_edit = QLineEdit()
        self.prompt_embeds_edit = QLineEdit()
        self.negative_prompt_embeds_edit = QLineEdit()
        self.output_type_combo = QComboBox()
        self.callback_edit = QLineEdit()
        self.image_width = QSpinBox()
        self.image_height = QSpinBox()
        self.grid_width = QSpinBox()
        self.grid_height = QSpinBox()
        self.update_button = QPushButton("Update Parameters")

        # Set default values for each parameter
        self.prompt_edit.setText("")
        self.image_edit.setText("")
        self.num_inference_steps_spin.setValue(100)
        self.guidance_scale_spin.setValue(7.5)
        self.image_guidance_scale_spin.setValue(1.5)
        self.negative_prompt_edit.setText("")
        self.num_images_per_prompt_spin.setValue(1)
        self.eta_spin.setValue(0.0)
        self.generator_edit.setText("")
        self.latents_edit.setText("")
        self.prompt_embeds_edit.setText("")
        self.negative_prompt_embeds_edit.setText("")
        self.output_type_combo.addItems(self.output_type_options)
        self.callback_edit.setText("")
        self.image_width.setMaximum(65536)
        self.image_width.setValue(128)
        self.image_height.setMaximum(65536)
        self.image_height.setValue(128)
        self.grid_width.setMaximum(65536)
        self.grid_width.setValue(2048)
        self.grid_height.setMaximum(65536)
        self.grid_height.setValue(2048)

        # Create a layout for each parameter control
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        prompt_layout.addWidget(self.prompt_edit)

        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Image:"))
        image_layout.addWidget(self.image_edit)

        num_inference_steps_layout = QHBoxLayout()
        num_inference_steps_layout.addWidget(QLabel("Number of Inference Steps:"))
        num_inference_steps_layout.addWidget(self.num_inference_steps_spin)

        guidance_scale_layout = QHBoxLayout()
        guidance_scale_layout.addWidget(QLabel("Guidance Scale:"))
        guidance_scale_layout.addWidget(self.guidance_scale_spin)

        image_guidance_scale_layout = QHBoxLayout()
        image_guidance_scale_layout.addWidget(QLabel("Image Guidance Scale:"))
        image_guidance_scale_layout.addWidget(self.image_guidance_scale_spin)

        negative_prompt_layout = QHBoxLayout()
        negative_prompt_layout.addWidget(QLabel("Negative Prompt:"))
        negative_prompt_layout.addWidget(self.negative_prompt_edit)

        num_images_per_prompt_layout = QHBoxLayout()
        num_images_per_prompt_layout.addWidget(QLabel("Number of Images per Prompt:"))
        num_images_per_prompt_layout.addWidget(self.num_images_per_prompt_spin)

        eta_layout = QHBoxLayout()
        eta_layout.addWidget(QLabel("Eta:"))
        eta_layout.addWidget(self.eta_spin)

        generator_layout = QHBoxLayout()
        generator_layout.addWidget(QLabel("Generator:"))
        generator_layout.addWidget(self.generator_edit)

        latents_layout = QHBoxLayout()
        latents_layout.addWidget(QLabel("Latents:"))
        latents_layout.addWidget(self.latents_edit)

        prompt_embeds_layout = QHBoxLayout()
        prompt_embeds_layout.addWidget(QLabel("Prompt Embeds:"))
        prompt_embeds_layout.addWidget(self.prompt_embeds_edit)

        negative_prompt_embeds_layout = QHBoxLayout()
        negative_prompt_embeds_layout.addWidget(QLabel("Negative Prompt Embeds:"))
        negative_prompt_embeds_layout.addWidget(self.negative_prompt_embeds_edit)

        output_type_layout = QHBoxLayout()
        output_type_layout.addWidget(QLabel("Output Type:"))
        output_type_layout.addWidget(self.output_type_combo)

        callback_layout = QHBoxLayout()
        callback_layout.addWidget(QLabel("Callback:"))
        callback_layout.addWidget(self.callback_edit)

        dimensions_layout = QHBoxLayout()
        dimensions_layout.addWidget(QLabel("Image, Canvas W/H:"))
        dimensions_layout.addWidget(self.image_width)
        dimensions_layout.addWidget(self.image_height)
        dimensions_layout.addWidget(self.grid_width)
        dimensions_layout.addWidget(self.grid_height)

        # Create a layout for the update button
        update_button_layout = QHBoxLayout()
        update_button_layout.addWidget(self.update_button)

        # Create a grid layout to hold all the parameter controls
        grid_layout = QGridLayout()
        grid_layout.addLayout(prompt_layout, 0, 0)
        grid_layout.addLayout(image_layout, 1, 0)
        grid_layout.addLayout(num_inference_steps_layout, 2, 0)
        grid_layout.addLayout(guidance_scale_layout, 3, 0)
        grid_layout.addLayout(image_guidance_scale_layout, 4, 0)
        grid_layout.addLayout(negative_prompt_layout, 5, 0)
        grid_layout.addLayout(num_images_per_prompt_layout, 6, 0)
        grid_layout.addLayout(eta_layout, 7, 0)
        grid_layout.addLayout(generator_layout, 8, 0)
        grid_layout.addLayout(dimensions_layout, 9, 0)

        grid_layout.addLayout(update_button_layout, 15, 0)

        # Connect the update button to a slot that emits the parameters_updated signal
        self.update_button.clicked.connect(self.emit_parameters_updated_signal)

        # Set the main layout of the widget
        self.setLayout(grid_layout)

    def emit_parameters_updated_signal(self):
        # Emit the parameters_updated signal with a dictionary of the current parameter values
        parameters = {
            "prompt": self.prompt_edit.text(),
            "image": self.image_edit.text(),
            "num_inference_steps": self.num_inference_steps_spin.value(),
            "guidance_scale": self.guidance_scale_spin.value(),
            "image_guidance_scale": self.image_guidance_scale_spin.value(),
            "negative_prompt": self.negative_prompt_edit.text(),
            "num_images_per_prompt": self.num_images_per_prompt_spin.value(),
            "eta": self.eta_spin.value(),
            "generator": self.generator_edit.text(),
            "latents": self.latents_edit.text(),
            "prompt_embeds": self.prompt_embeds_edit.text(),
            "negative_prompt_embeds": self.negative_prompt_embeds_edit.text(),
            "output_type": self.output_type_combo.currentText(),
            "callback": self.callback_edit.text(),
            "callback_steps": self.callback_steps_spin.value()
        }
        self.parameters_updated.emit(parameters)
class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, lock=False, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.lock = lock
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())