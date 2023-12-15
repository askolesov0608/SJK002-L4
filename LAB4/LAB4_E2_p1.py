import cv2
from skimage import feature
import matplotlib.pyplot as plt

def testCanny(im, sigma=1.0, low_threshold_ratio=0.1, high_threshold_ratio=0.2): #use_quantiles установлен в False T1-T2 0-255 True pers 0.1-1
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
    edges = feature.canny(im, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=False)

    return edges

# Чтение и преобразование изображения в оттенки серого
image = cv2.imread('./Test/1.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Ошибка: Не удалось загрузить изображение. Проверьте путь к файлу.")
    exit()
    
    # Изменение размера до 128x128
image_resized = cv2.resize(image, (128, 128))

# Проверка количества каналов в изображении и преобразование в оттенки серого
if len(image_resized.shape) == 3:  # Проверка на наличие 3 каналов (BGR)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) / 255.0
else:
    image_gray = image_resized / 255.0

# Применение testCanny
edges = testCanny(image, sigma=3, low_threshold_ratio=0.01, high_threshold_ratio=0.02)

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
