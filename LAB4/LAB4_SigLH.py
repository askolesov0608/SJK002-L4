import cv2
from skimage import feature
import matplotlib.pyplot as plt

def testCanny(im, sigma=1.0, low_threshold_ratio=0.2, high_threshold_ratio=0.25):
    """
    Применяет детектор Канни к изображению.

    :param im: Изображение в оттенках серого.
    :param sigma: Стандартное отклонение для Гауссова размытия.
    :param low_threshold_ratio: Нижний порог для детектора Канни.
    :param high_threshold_ratio: Верхний порог для детектора Канни.
    :return: Изображение с результатами детектора Канни.
    """
    # Преобразование порогов в абсолютные значения
    low_threshold = low_threshold_ratio * 255
    high_threshold = high_threshold_ratio * 255

    # Применение детектора Канни
    edges = feature.canny(im, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    return edges

# Чтение и преобразование изображения в оттенки серого
image = cv2.imread('./Test/1.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Ошибка: Не удалось загрузить изображение. Проверьте путь к файлу.")
    exit()

# Применение testCanny
edges = testCanny(image, sigma=1, low_threshold_ratio=0.4, high_threshold_ratio=0.6)

# Отображение результатов
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges Detected')
plt.axis('off')

plt.show()
