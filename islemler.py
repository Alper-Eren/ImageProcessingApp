import cv2
import numpy as np
from PyQt5.QtGui import QImage

class Methods(object):
    #TODO gri yap

    def convert_array_to_qimage(self, array):
        height, width = array.shape
        bytes_per_line = width
        array_uint8 = np.array(array, dtype=np.uint8, order='C')  # Yeni düzenleme
        q_image = QImage(array_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        return q_image
    
    def fourier_transform(self, image_array):
        
        # image_array = np.array(image_array, dtype=np.uint8, order='C')  # Yeni düzenleme
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        # Fourier dönüşümü uygulama
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.uint8(20 * np.log(np.abs(f_transform_shifted)))
        q_image = self.convert_array_to_qimage(magnitude_spectrum)
        current_image = q_image.copy()
        return current_image

    def sobel_filter(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  
        # Sobel filtresini uygulama
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # Kenar bilgisini birleştirme
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
        current_image = sobel_combined.copy()
        return current_image

    def histogram(self, image_array, slider_value=0.5):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)        
        # Görüntüyü eşikleme işlemine tabi tutma
        threshold_value = int(slider_value * 255)  # Slider değerini 0-255 aralığına dönüştürelim
        _, threshold_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        current_image = threshold_image.copy()
        return current_image

    def opening(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  
        # Kernel boyutunu belirle (örneğin 3x3 bir matris)
        kernel_size = (3, 3)
        # Erozyon işlemi uygula
        erosion = cv2.erode(gray, kernel_size, iterations=1)
        # Genişleme işlemi uygula
        dilation = cv2.dilate(erosion, kernel_size, iterations=1)
        current_image = dilation.copy()
        return current_image

    def closing(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  
        kernel = np.ones((5, 5), np.uint8)
        # Kapanış işlemi uygulama
        closing_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        current_image = closing_image.copy()
        return current_image

    def dilation(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  
        kernel = np.ones((5, 5), np.uint8)
        # Genişleme işlemi uygulama
        dilation_image = cv2.dilate(gray, kernel, iterations=1)
        current_image = dilation_image.copy()
        return current_image

    def erosion(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  
        kernel = np.ones((5, 5), np.uint8)
        # Aşındırma işlemi uygulama
        erosion_image = cv2.erode(gray, kernel, iterations=1)
        current_image = erosion_image.copy()
        return current_image

    def low_pass_filter(self, image_array, slider_value=0.5):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        radius = int(min(center_row, center_col) * slider_value)
        # Fourier dönüşümünü yapın
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        # Düşük frekans bileşenleri koruyun
        f_transform_shifted[center_row - radius: center_row + radius + 1, center_col - radius: center_col + radius + 1] = 0
        # Ters Fourier dönüşümünü yapın
        f_transform_inverse_shifted = np.fft.ifftshift(f_transform_shifted)
        image_filtered = np.fft.ifft2(f_transform_inverse_shifted)
        image_filtered = np.abs(image_filtered)  # Extract the real part
        
        q_image = self.convert_array_to_qimage(image_filtered)
        current_image = q_image.copy()
        return current_image

    def high_pass_filter(self, image_array, slider_value=0.5):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        # Yüksek geçiren filtre için cutoff frekansı hesaplayın
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        radius = int(min(center_row, center_col) * slider_value)
        # Fourier dönüşümünü yapın
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        # Yüksek frekans bileşenleri koruyun
        f_transform_shifted[center_row - radius: center_row + radius + 1, center_col - radius: center_col + radius + 1] = f_transform[center_row - radius: center_row + radius + 1, center_col - radius: center_col + radius + 1]
        # Ters Fourier dönüşümünü yapın
        f_transform_inverse_shifted = np.fft.ifftshift(f_transform_shifted)
        image_filtered = np.fft.ifft2(f_transform_inverse_shifted)
        image_filtered = np.abs(image_filtered)
        q_image = self.convert_array_to_qimage(image_filtered)
        current_image = q_image.copy()
        return current_image