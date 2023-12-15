from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import glob
import sys
from scipy.ndimage import sobel
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough_line
from scipy import ndimage as ndi
from copy import deepcopy

#sys.path.append("../../p1/code")
import visualPercepUtilsModx3 as vpu

bLecturerVersion=False
# try:
#     import p4e
#     bLecturerVersion=True
# except ImportError:
#     pass # file only available to lecturers

def testSobel(im, params=None):
    gx = filters.sobel(im, 1) #axis 0 corresponds to the vertical direction (y-axis), and axis 1 corresponds to the horizontal direction (x-axis).
    gy = filters.sobel(im, 0)
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = np.mean(magnitude)  # This is a simple threshold;
    binary_edge = (magnitude > threshold).astype(np.float64)
    return [gx, gy, magnitude, binary_edge]
    """ Входные параметры:
                        im: Входное изображение (предполагается, что оно в градациях серого).
                        nbins: Количество бинов для гистограммы ориентированных градиентов.
        Вычисление градиентов:
                         Функция использует оператор Собеля для вычисления градиентов по горизонтальной (gx) и вертикальной (gy) осям.
        Вычисление величины и направления градиентов:
                          Величина градиента (magnitude) вычисляется как корень из суммы квадратов компонент gx и gy.
                          Направление градиента (orientation) вычисляется с использованием функции arctan2, 
                          результат переводится из радиан в градусы.
        Создание гистограммы:
                          С помощью функции np.histogram создается гистограмма ориентированных градиентов
                          с заданным количеством бинов, охватывающих диапазон от 0 до 180 градусов.
                          Веса для гистограммы определяются величиной градиента. """
def HOG(im, nbins):
    """
    Compute the Histogram of Oriented Gradients (HOG) for an image.

    :param im: Input image (assumed to be grayscale).
    :param nbins: Number of bins for the histogram of gradient orientations.
    :return: Histogram of oriented gradients.
    """
    
    # Compute gradients along the x and y axes
    gx = sobel(im, axis=1)  # horizontal gradient
    gy = sobel(im, axis=0)  # vertical gradient

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)  # convert from radians to degrees
    orientation[orientation < 0] += 180  # convert from [-180, 180] to [0, 180] range

    # Create histogram of oriented gradients
    hog, _ = np.histogram(orientation, bins=nbins, range=(0, 180), weights=magnitude)

    return hog


Для реализации функции displayLines(), которая будет рисовать линии на изображении на основе обнаруженных пиков в пространстве Хафа, мы должны вычислить координаты двух точек на каждой линии и затем нарисовать эти линии. Функция будет принимать изображение и параметры пиков, обнаруженных в результате трансформации Хафа.

Вот пример реализации функции displayLines():

python
Copy code
import matplotlib.pyplot as plt
import numpy as np

def displayLines(image, thetas, rhos):
    """
    Рисует линии на изображении на основе обнаруженных пиков в пространстве Хафа.
    
    :param image: Входное изображение.
    :param thetas: Углы θ пиков в радианах.
    :param rhos: Расстояния ρ пиков.
    """
    H, W = image.shape
    for theta, rho in zip(thetas, rhos):
        # Переводим theta из радианов в градусы для удобства
        theta_deg = np.rad2deg(theta)

        # Вычисляем две точки на каждой линии
        if np.sin(theta) != 0:
            # y = (rho - x*cos(theta)) / sin(theta)
            y1 = (rho - 0 * np.cos(theta)) / np.sin(theta)
            y2 = (rho - W * np.cos(theta)) / np.sin(theta)
            plt.plot((0, W), (y1, y2), '-r')
        else:
            # x = rho / cos(theta)
            x = rho / np.cos(theta)
            plt.plot((x, x), (0, H), '-r')

    # Отображаем изображение с линиями
    plt.imshow(image, cmap='gray')
    plt.title("Lines detected")
    plt.axis('off')
    plt.show()

def testCanny(im, params=None):
    sigma = params['sigma']
    edge = feature.canny(im, sigma=sigma, low_threshold=0.2 * 255, high_threshold=0.25 * 255, use_quantiles=False)
    return [edge]


def testHough(im, params=None):
    edges = testCanny(im, params)[0]
    numThetas = 200
    H, thetas, rhos = hough_line(edges, np.linspace(-np.pi/2, np.pi/2, numThetas))
    print("# angles:", len(thetas))
    print("# distances:", len(rhos))
    print("rho[...]",rhos[:5],rhos[-5:])
    return [np.log(H+1), (H, thetas, rhos)] # log of Hough space for display purpose


def findPeaks(H, thetas, rhos, nPeaksMax=None):
    if nPeaksMax is None:
        nPeaksMax = np.inf
    return hough_line_peaks(H, thetas, rhos, num_peaks=nPeaksMax, threshold=0.15 * np.max(H), min_angle=20, min_distance=15)


# -----------------
# Test image files
# -----------------
path_input = './imgs-P4/'
path_output = './imgs-out-P4/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'cuadros.png']  # cuadros, lena

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testSobel', 'testCanny', 'testHough']
else:
    tests = ['testSobel']
    tests = ['testCanny']
    tests = ['testHough']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testSobel': 'Detector de Sobel',
             'testCanny': 'Detector de Canny',
             'testHough': 'Transformada de Hough'}

bAddNoise = True
bRotate = False


def doTests():
    print("Testing on", files)
    nFiles = len(files)
    nFig = None
    for test in tests:
        if test is "testSobel":
            params = {}
        elif test in ["testCanny", "testHough"]:
            params = {}
            params['sigma'] = 5  # 15
        
        if test is "testHough":
            pass  # params={}

        for i, imfile in enumerate(files):
            print("testing", test, "on", imfile)

            im_pil = Image.open(imfile).convert('L')
            im = np.array(im_pil)  # from Image to array

            if bRotate:
                im = ndi.rotate(im, 90, mode='nearest')

            if bAddNoise:
                im = im + np.random.normal(loc=0, scale=5, size=im.shape)

            outs_np = eval(test)(im, params)
            print("num ouputs", len(outs_np))
            
            if test is "testHough":
                outs_np_plot = outs_np[0:1]
            else:
                outs_np_plot = outs_np
            nFig = vpu.showInFigs([im] + outs_np_plot, title=nameTests[test], nFig=nFig, bDisplay=True)  # bDisplay=True for displaying *now* and waiting for user to close
            
            # Добавление функции HOG
            if test == "testSobel":  # Пример использования HOG с результатом testSobel
                nbins = 9  # Количество бинов для HOG
                hog = HOG(im, nbins)
                plt.figure()
                plt.bar(range(nbins), hog)
                plt.title("HOG Histogram of Oriented Gradients")
                plt.show()

            if test is "testHough":
                H, thetas, rhos = outs_np[1]  # second output is not directly displayable
                peaks_values, peaks_thetas, peaks_rhos = findPeaks(H, thetas, rhos, nPeaksMax=None)
                displayLines(im, peaks_thetas, peaks_rhos)
                vpu.displayHoughPeaks(H, peaks_values, peaks_thetas, peaks_rhos, thetas, rhos)
                if bLecturerVersion:
                    p4e.displayLines(im, peaks_thetas, peaks_rhos, peaks_values) # exercise
                    plt.show(block=True)
                # displayLineSegments(...) # optional exercise

    plt.show(block=True)  # show pending plots (useful if we used bDisplay=False in vpu.showInFigs())


if __name__ == "__main__":
    doTests()

