import gzip
import os
import pickle
import sys
import time
from typing import Tuple, List
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from huffman import huffman_decode, huffman_encode, build_huffman_tree
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
from determine_frames_types import get_frame_types
from utils import print_progress_bar, save_encoded_frames, compare_folder_and_file_size, save_decoded_frames, \
    sec2timestr, rgb2yuv, yuv2rgb, quant_matrix_I, quant_matrix_PB
from length_coding import run_length_encode, run_length_decode



def unpack_from_file(file_path: str):
    """
    Распаковать данные из файла.

    :param file_path: Путь к бинарному файлу
    :return: Кортеж с распакованными данными
    """
    # Чтение данных из файла
    with open(file_path, 'rb') as file:
        compressed_data = file.read()
    #
    # Распаковка данных
    serialized_data = gzip.decompress(compressed_data)
    data_loaded = pickle.loads(serialized_data)

    return data_loaded



def decode_frames(mpeg):
    """
    Декодирует MPEG поток в видеоролик.

    Аргументы:
    mpeg -- список фреймов, каждый из которых закодирован и представлен как словарь или объект.

    Возвращает:
    mov -- декодированный видеоролик в формате RGB.
    """
    # Получаем размеры первого фрейма
    movsize = mpeg[0].shape
    mov = np.zeros((16*movsize[0], 16*movsize[1], 3, len(mpeg)), dtype=np.uint8)

    # Инициализация предыдущего фрейма
    pf = None

    # Декодирование каждого фрейма
    for i in range(len(mpeg)):
        # Декодирование фрейма
        f = decode_frame(mpeg[i], pf)

        # Сохранение предыдущего фрейма
        pf = f.copy()

        # Конвертация фрейма из YCbCr в RGB
        f = yuv2rgb(f)

        # Проверка, что значения пикселей находятся в диапазоне [0, 255]
        f = np.clip(f, 0, 255)

        # Сохранение фрейма в массив видеоролика
        mov[:, :, :, i] = f.astype(np.uint8)

        print_progress_bar((i+1) * 100 / len(mpeg), 100, prefix='Происходит декодирование кадров:', suffix='Готово', length=50)

    return mov

def decode_frame(mpeg, pf):
    """
    Декодирует кадр из MPEG потока.

    Аргументы:
    mpeg -- массив объектов или словарей, каждый из которых содержит информацию для одного макроблока.
    pf -- предыдущий кадр, используемый для предсказания в P-кадрах.

    Возвращает:
    fr -- декодированный кадр в виде массива.
    """
    # Размеры массива макроблоков
    blocksize = mpeg.shape
    M = 16 * blocksize[0]
    N = 16 * blocksize[1]

    # Инициализируем кадр нулями
    fr = np.zeros((M, N, 3))

    # Перебор макроблоков
    for m in range(blocksize[0]):
        for n in range(blocksize[1]):
            # Вычисляем координаты в кадре для текущего макроблока
            x = slice(16 * m, 16 * (m + 1))
            y = slice(16 * n, 16 * (n + 1))
            # Декодирование макроблока и помещение его в кадр
            fr[x, y, :] = decode_block(mpeg[m, n], pf, 16 * m, 16 * n)

    return fr


def decode_block(mpeg, pf, x, y):
    """
    Декодирование макроблока из MPEG потока.

    Аргументы:
    mpeg -- словарь или объект с данными MPEG (тип кадра, векторы движения, коэффициенты, масштабы).
    pf -- предыдущий кадр (для использования в предсказании).
    x, y -- координаты начала макроблока в предыдущем кадре.

    Возвращает:
    block -- декодированный макроблок.
    """
    # Инициализация матриц квантования
    q1, q2 = quant_matrix_I(), quant_matrix_PB()[1]  # Предположим, что qinter возвращает матрицу

    block = np.zeros((16, 16, 3))

    # Предсказание с использованием векторов движения
    if mpeg['type'] == 'P':
        # Получаем предсказанный блок из предыдущего кадра
        predicted_block = pf[x + mpeg['mvx']: x + mpeg['mvx'] + 16, y + mpeg['mvy']: y + mpeg['mvy'] + 16, :]
        # Обеспечиваем, что размеры совпадают
        block[:predicted_block.shape[0], :predicted_block.shape[1], :] = predicted_block
        q = q2
    else:
        q = q1

    # Декодирование блоков
    b = np.zeros((8, 8, 6))
    for i in range(6):
        # Декодируем Хаффман для получения RLE кодов
        decoded_rle = huffman_decode(mpeg['huffman'][i], mpeg['huffman_tree'][i])
        if not isinstance(decoded_rle, list) or not all(isinstance(x, tuple) for x in decoded_rle):
            raise ValueError("decoded_rle должен быть списком кортежей")
        # Применяем обратное RLE
        decoded_values = run_length_decode(decoded_rle)
        # Преобразуем список обратно в матрицу 8x8
        b[:, :, i] = np.array(decoded_values).reshape(8, 8)
        # Применяем обратное DCT
        coef = b[:, :, i] * (mpeg['scale'][i] * q) / 8
        b[:, :, i] = cv2.idct(coef)
        # b[:, :, i] = idct(coef)


    # Конструкция макроблока
    block += putblocks(b)

    return block

def putblocks(b):
    """
    Собирает блоки DCT в один макроблок.

    Аргументы:
    b -- массив блоков DCT, где b[:,:,i] представляет i-й блок.

    Возвращает:
    block -- макроблок собранный из входных блоков.
    """
    # Создаем пустой макроблок
    block = np.zeros((16, 16, 3))

    # Заполняем яркостные блоки
    for i in range(4):  # Для четырех блоков яркости
        row = (i // 2) * 8
        col = (i % 2) * 8
        if b.shape[2] > i:  # Проверяем, что блок существует
            block[row:row+8, col:col+8, 0] = b[:, :, i]

    # Заполняем блоки цветности
    if b.shape[2] > 4:  # Кронекерово произведение для блока Cb
        block[:, :, 1] = np.kron(b[:, :, 4], np.ones((2, 2)))
    if b.shape[2] > 5:  # Кронекерово произведение для блока Cr
        block[:, :, 2] = np.kron(b[:, :, 5], np.ones((2, 2)))

    return block


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет PSNR между двумя изображениями. Обрезает первое изображение до размеров второго, если необходимо.

    :param img1: Первое изображение.
    :param img2: Второе изображение.
    :return: PSNR между двумя изображениями.
    """
    # Обрезаем img1 до размеров img2, если их размеры не совпадают
    if img1.shape != img2.shape:
        img1 = img1[:img2.shape[0], :img2.shape[1]]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr


def calculate_quality_degradation(original_image_path: str, compressed_image_path: str, baseline_image_path: str):
    """
    Вычисляет процент деградации качества между изображениями.

    :param original_image_path: Путь к оригинальному изображению.
    :param compressed_image_path: Путь к сжатому изображению.
    :param baseline_image_path: Путь к базовому изображению.
    :return: SSIM с сжатым изображением, Процент деградации качества.
    """
    # Загрузка изображений
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    compressed_image = cv2.imread(compressed_image_path, cv2.IMREAD_GRAYSCALE)
    baseline_image = cv2.imread(baseline_image_path, cv2.IMREAD_GRAYSCALE)

    # Убедиться, что все изображения успешно загружены
    if original_image is None or compressed_image is None or baseline_image is None:
        raise ValueError("Одно или несколько изображений не удалось загрузить.")

    # Определение минимальных размеров для обрезки
    min_height = min(original_image.shape[0], compressed_image.shape[0], baseline_image.shape[0])
    min_width = min(original_image.shape[1], compressed_image.shape[1], baseline_image.shape[1])

    # Обрезка изображений
    original_image = original_image[:min_height, :min_width]
    compressed_image = compressed_image[:min_height, :min_width]
    baseline_image = baseline_image[:min_height, :min_width]

    # Вычисление SSIM
    ssim_compressed = ssim(original_image, compressed_image)
    ssim_baseline = ssim(original_image, baseline_image)

    # Вычисление процентной деградации качества
    quality_degradation = ((ssim_baseline - ssim_compressed) / ssim_baseline) * 100

    return ssim_compressed, quality_degradation


def process_sequence_psnr_and_degradation(original_folder: str, encoded_folder: str, q90_folder: str) -> Tuple[float, List[float], float, List[float]]:
    """
    Вычисляет PSNR и процент деградации между изображениями в указанных директориях.

    :param original_folder: Папка с исходными изображениями.
    :param encoded_folder: Папка с закодированными изображениями.
    :param q90_folder: Папка с изображениями с качеством 90%
    :return: Среднее значение PSNR, список значений PSNR, среднее значение процентной деградации и список процентов деградации для каждого изображения.
    """
    original_files = sorted(os.listdir(original_folder))
    encoded_files = sorted(os.listdir(encoded_folder))
    q90_files = sorted(os.listdir(q90_folder))

    psnr_list = []
    ssim_list = []
    degradation_list = []

    table_data = []

    for orig_file, enc_file, q90_file in zip(original_files, encoded_files, q90_files):
        orig_path = os.path.join(original_folder, orig_file)
        enc_path = os.path.join(encoded_folder, enc_file)
        q90_path = os.path.join(q90_folder, q90_file)

        # Загрузка изображений
        orig_img = cv2.imread(orig_path)
        enc_img = cv2.imread(enc_path)
        q90_img = cv2.imread(q90_path)

        if orig_img is None or enc_img is None or q90_img is None:
            continue

        # Вычисление PSNR
        psnr = calculate_psnr(orig_img, enc_img)
        psnr_list.append(psnr)

        ssim, degradation = calculate_quality_degradation(orig_path, enc_path, q90_path)
        ssim_list.append(ssim)
        degradation_list.append(degradation)
        table_data.append([orig_file, enc_file, f"{psnr:.2f}", f"{ssim:.2f}", f"{degradation:.2f}%"])

    # Средние значения PSNR и процентной деградации
    if psnr_list and degradation_list and ssim_list:
        average_psnr = sum(psnr_list) / len(psnr_list)
        average_degradation = sum(degradation_list) / len(degradation_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        table_data.append(["Среднее", "", f"{average_psnr:.2f}", f"{average_ssim:.2f}", f"{average_degradation:.2f}%"])

        headers = ["Оригинальное изображение", "Декодированное изображение", "PSNR", "SSIM", "Коэффициент деградации"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        sys.stderr.write(table)

        return average_psnr, psnr_list, average_degradation, degradation_list, average_ssim, ssim_list
    else:
        sys.stderr.write("Нет валидных изображений в директориях. \n")
        return None, [], None, []

if __name__ == '__main__':

    sys.stderr.write("\nРеализация декодирования MPEG2 by DB\n")

    mpeg = unpack_from_file('data.bin')

    # Decoding
    start_time = time.time()
    mov2 = decode_frames(mpeg)
    decode_time = time.time() - start_time
    sys.stderr.write(f'\nОбщее время декодирования: {sec2timestr(decode_time)}\n')

    decoded_folder = 'data/decoded_frames/'
    save_decoded_frames(mov2, decoded_folder)

    sys.stderr.write(f"Начинаю вычислять PSNR, SSIM и процент деградации между оригинальными и декодированными изображениями: \n")
    encoded_folder = 'data/encoded_frames/'
    decoded_folder = 'data/decoded_frames/'
    q90_folder = 'data/frames_Q90/'
    # Вычисление PSNR и коэффициента деградации
    average_psnr, psnr_list, average_degradation, degradation_list, average_ssim, ssim_list = process_sequence_psnr_and_degradation(encoded_folder, decoded_folder, q90_folder)