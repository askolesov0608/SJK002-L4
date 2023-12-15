import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from skimage import feature, exposure, filters
from skimage.transform import hough_line, hough_line_peaks
from scipy import ndimage as ndi
from PIL import Image


def adaptiveCanny(image, sigma=2.5, low_threshold_ratio=0.1, high_threshold_ratio=0.2):
    # Улучшаем контрастность изображения
    image_eq = exposure.equalize_adapthist(image)

    # Регулируем пороги для алгоритма Канни
    low_threshold = low_threshold_ratio * np.max(image_eq)
    high_threshold = high_threshold_ratio * np.max(image_eq)
    
    edges = feature.canny(image_eq, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return edges


def testCanny(im, sigma=2.5):
    edge = feature.canny(im, sigma=sigma)
    return edge

def displayLines(image, thetas, rhos):
    H, W = image.shape
    plt.imshow(image, cmap='gray')

    for theta, rho in zip(thetas, rhos):
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = rho * a, rho * b
        x1, y1 = x0 + W * (-b), y0 + W * a
        x2, y2 = x0 - W * (-b), y0 - W * a
        x1, y1 = max(min(x1, W), 0), max(min(y1, H), 0)
        x2, y2 = max(min(x2, W), 0), max(min(y2, H), 0)
        plt.plot((x1, x2), (y1, y2), '-r')

    plt.title("Lines detected")
    plt.axis('off')
    plt.show()

def findLineSegments(image):
    edges = adaptiveCanny(image)

    # Применяем преобразование Хафа с измененными параметрами
    H, thetas, rhos = hough_line(edges)
    _, angles, distances = hough_line_peaks(H, thetas, rhos, threshold=0.2 * np.max(H), min_distance=20, min_angle=20)

    displayLines(image, angles, distances)

# Загрузка изображения
im_path = './Test2/cuadros.png'  # Замените на путь к вашему изображению
im = np.array(Image.open(im_path).convert('L'))

# Обнаружение сегментов прямых линий
findLineSegments(im)
