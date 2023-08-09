import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem, QSlider
from PyQt5.QtCore import Qt
from app import Ui_MainWindow
from islemler import Methods
import PyQt5.QtGui as QtGui
import cv2
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.methods = Methods()
        # Bağlantıları yapılandırma
        self.choose_image_button.clicked.connect(self.choose_image)
        self.choose_file_button.clicked.connect(self.choose_file)
        self.file_list_widget.itemClicked.connect(self.show_selected_image)
        
        self.comboBox.activated.connect(self.apply_selected_filter)
        self.save_button.clicked.connect(self.export_result)
        self.arrow_back_button.clicked.connect(self.transfer)
        
        self.selected_image_path = None
        self.is_selected_image = False  
        self.current_processed_image = None

      
        self.slider_value = 0.5  # Slider değeri için özellik tanımlama
        self.populate_filter_combobox()
        # Slider oluşturma ve özelliklerini belirleme
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)  # Varsayılan değer
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.apply_selected_filter)
        self.slider.valueChanged.connect(self.update_slider_label)  # Yeni bağlantı

    def choose_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose an image file", "", "Image Files (*.jpg; *.jpeg; *.png);;All Files (*)", options=options)
        if file_name:
            self.selected_image_path = file_name
            self.is_selected_image = True
            self.image = cv2.imread(self.selected_image_path)
            self.load_and_show_image()  

    def choose_file(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select a directory", "", options=options)
        if directory:
            self.is_selected_image = False  # Image seçimi iptal edildi
            self.file_list_widget.clear()
            supported_formats = ["jpg", "jpeg", "png"]
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if filename.lower().endswith(tuple(supported_formats)) and os.path.isfile(file_path):
                    item = QtWidgets.QListWidgetItem(filename)
                    item.setData(Qt.UserRole, file_path)
                    self.file_list_widget.addItem(item)
   
    def show_selected_image(self, item):
        selected_file = item.data(Qt.UserRole)
        self.selected_image_path = selected_file
        self.is_selected_image = True
        self.image = cv2.imread(self.selected_image_path)
        self.load_and_show_image()
        self.image_result_label.clear()

        # print("Image Array:")
        # print(self.image)
   
    def load_and_show_image(self):
        if self.is_selected_image:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(self.image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.clear()

    def apply_image_filter(self, filter_func, image_array, current_image=None, slider_value=None):
        if current_image is None:
            current_image = image_array.copy()  # Orijinal görüntüyü kopyalayın
        
        if slider_value is not None:
            filtered_image = filter_func(current_image, slider_value=slider_value)
        else:
            filtered_image = filter_func(current_image)
            
        if filtered_image is not None:
            self.show_result(filtered_image)

    def apply_selected_filter(self):
        selected_filter = self.comboBox.currentText()
        slider_value = self.slider.value() / 100.0  # Slider değerini 0-1 aralığına dönüştürelim

        # Slider'ın durumunu seçilen filtreye göre güncelle
        if selected_filter in ["Histogram", "Low Pass Filter", "High Pass Filter"]:
            self.slider.setEnabled(True)
        else:
            self.slider.setEnabled(False)

        if self.is_selected_image:
            if selected_filter == "Fourier Transform":
                self.apply_image_filter(self.methods.fourier_transform, image_array=self.image, slider_value=None)
            elif selected_filter == "Sobel Filter":
                self.apply_image_filter(self.methods.sobel_filter, image_array=self.image, slider_value=None)
            elif selected_filter == "Histogram":
                self.apply_image_filter(self.methods.histogram, image_array=self.image, slider_value=slider_value)
            elif selected_filter == "Opening":
                self.apply_image_filter(self.methods.opening, image_array=self.image, slider_value=None)
            elif selected_filter == "Closing":
                self.apply_image_filter(self.methods.closing, image_array=self.image, slider_value=None)
            elif selected_filter == "Dilation":
                self.apply_image_filter(self.methods.dilation, image_array=self.image, slider_value=None)
            elif selected_filter == "Erosion":
                self.apply_image_filter(self.methods.erosion, image_array=self.image, slider_value=None)
            elif selected_filter == "Low Pass Filter":
                self.apply_image_filter(self.methods.low_pass_filter, image_array=self.image, slider_value=slider_value)
            elif selected_filter == "High Pass Filter":
                self.apply_image_filter(self.methods.high_pass_filter, image_array=self.image, slider_value=slider_value)
        else:
            # Dosya listesinde seçili olan resmin yolunu alıp işleme gönder
            selected_item = self.file_list_widget.currentItem()
            if selected_item:
                image_path = selected_item.data(Qt.UserRole)
                image_array = cv2.imread(image_path)
                if selected_filter == "Fourier Transform":
                    self.apply_image_filter(self.methods.fourier_transform, image_array)
                elif selected_filter == "Sobel Filter":
                    self.apply_image_filter(self.methods.sobel_filter, image_array)
                elif selected_filter == "Histogram":
                    self.apply_image_filter(self.methods.histogram, image_array=image_array, slider_value=slider_value)
                elif selected_filter == "Opening":
                    self.apply_image_filter(self.methods.opening, image_array)
                elif selected_filter == "Closing":
                    self.apply_image_filter(self.methods.closing, image_array)
                elif selected_filter == "Dilation":
                    self.apply_image_filter(self.methods.dilation, image_array)
                elif selected_filter == "Erosion":
                    self.apply_image_filter(self.methods.erosion, image_array)
                elif selected_filter == "Low Pass Filter":
                    self.apply_image_filter(self.methods.low_pass_filter, image_array=image_array, slider_value=slider_value)
                elif selected_filter == "High Pass Filter":
                    self.apply_image_filter(self.methods.high_pass_filter, image_array=image_array, slider_value=slider_value)

    def update_slider_label(self, value):
        self.slider_label.setText(f"Slider Value:{value}")

    def show_result(self, image_array):
        if isinstance(image_array, np.ndarray):  # Check if it's a NumPy array
            if len(image_array.shape) == 2:
                # Grayscale image, convert to RGB format
                image_array_rgb = np.stack([image_array] * 3, axis=-1)
            else:
                image_array_rgb = image_array  # Already in RGB format
            height, width, _ = image_array_rgb.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image_array_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        else:  # It's already a QImage
            q_image = image_array
        
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.image_result_label.setPixmap(pixmap.scaled(self.image_result_label.size(), Qt.KeepAspectRatio))

        
    def transfer(self):
        image_result_label_pixmap = self.image_result_label.pixmap()
        if image_result_label_pixmap!=None:
            # image_label'a image_result_label'daki görüntüyü aktar
            self.image_label.setPixmap(image_result_label_pixmap)
            # image_result_label'ı boşaltma
            self.image_result_label.clear()
        else:
            pass

    def export_result(self):
        pixmap = self.image_result_label.pixmap()
        if pixmap:
            q_image = pixmap.toImage()
            if not q_image.isNull():
                options = QFileDialog.Options()
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg; *.jpeg);;All Files (*)", options=options)
                if file_name:
                    q_image.save(file_name)

    def populate_filter_combobox(self):
        # İşlem adlarını ComboBox'a ekleyin
        filters = [
            "Choose Process",
            "Fourier Transform",
            "Sobel Filter",
            "Histogram",
            "Opening",
            "Closing",
            "Dilation",
            "Erosion",
            "Low Pass Filter",
            "High Pass Filter"
        ]
        self.comboBox.addItems(filters)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
