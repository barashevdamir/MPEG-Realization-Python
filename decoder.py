import ast
import gzip
import math
import pickle
import sys
from typing import Tuple, List
from tabulate import tabulate
import imageio.v2 as imageio
import cv2
from huffman import huffman_decoding
import os
import numpy as np
from utils import print_progress_bar
from skimage import io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from utils import zigzag_order, zigzag_index


def median_filter(input_frame: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Применяет фильтр медианного ядра к входному изображению.

    :param input_frame: Входящее изображение
    :param kernel_size: Размер ядра
    :return: Измененное изображение
    """
    # Размеры входного массива
    height, width = input_frame.shape[:2]
    # Половина размера ядра для удобства
    k = kernel_size // 2
    # Создание копии массива для вывода результата
    output_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Проход по всем пикселям изображения, кроме краев, которые не обрабатываются
    for i in range(k, height - k):
        for j in range(k, width - k):
            # Извлечение окна вокруг пикселя
            window = input_frame[i - k: i + k + 1, j - k: j + k + 1]
            # Вычисление медианы для каждого канала цвета
            median = np.median(window, axis=(0, 1))
            # Запись медианы в центр окна в результате
            output_frame[i, j] = median

    return output_frame

def convert_frames_to_rgb(yuv_frames: list) -> list:
    """
    Конвертирует кадры из формата YUV в RGB.

    :param yuv_frames: Список кадров в формате YUV
    :return: Список кадров в формате RGB
    """
    # Обратная матрица преобразования
    inverse_transform_matrix = np.linalg.inv(np.array([
        [0.229, 0.587, 0.144],
        [0.5, -0.4187, -0.0813],
        [0.1687, -0.3313, 0.5]
    ]))
    offset = np.array([0, 128, 128])
    rgb_frames = []
    for yuv_frame in yuv_frames:
        # Применяем обратное преобразование
        normalized_frame = yuv_frame.astype(np.float32) - offset
        rgb_frame = np.dot(normalized_frame.reshape(-1, 3), inverse_transform_matrix.T)
        rgb_frame = np.clip(rgb_frame, 0, 255)
        rgb_frame = rgb_frame.reshape(yuv_frame.shape).astype(np.uint8)

        rgb_frames.append(rgb_frame)

    return rgb_frames


def pad_block_to_8x8(block: np.ndarray) -> np.ndarray:
    """
    Дополняет блок до размера 8x8, используя нулевое заполнение.

    :param block: Блок для дополнения
    :return: Блок дополненный до 8x8
    """
    if block.shape == (8, 8):
        return block
    padded_block = np.zeros((8, 8), dtype=block.dtype)
    padded_block[:block.shape[0], :block.shape[1]] = block
    return padded_block


def run_length_decode(arr: list) -> list:
    """
    Применяет декодирование по длинам серий к 1D массиву.

    :param arr: 1D массив
    :return: Список элементов, полученных RLE-декодированием
    """
    result = []
    for element in arr:
        value, count = ast.literal_eval(element)
        result.extend([value]*count)
    return result

def zigzag_scan_inverse(arr: list) -> np.ndarray:
    """
    Преобразует одномерный массив обратно в матрицу 8x8.

    :param arr: 1D массив
    :return: Матрица 8x8
    """
    size = 8
    if len(arr) != size**2:
        raise ValueError("Длина массива не соответствует ожидаемой для матрицы 8x8.")
    matrix = np.zeros((size, size), dtype=float)
    index = 0
    for d in range(2 * size - 1):
        for i in range(max(0, d - size + 1), min(size, d + 1)):
            j = d - i
            if i < size and j < size:
                if d % 2 == 0:
                    matrix[j, i] = arr[index]
                else:
                    matrix[i, j] = arr[index]
                index += 1
    return matrix

def idct(dct_block: np.ndarray) -> np.ndarray:
    """
    Вычисляет двумерное обратное дискретное косинусное преобразование (IDCT) для данного блока.
    IDCT преобразует блок из частотной области обратно в пространственную.

    param dct_block: Входящий блок
    return: Выходной блок

    """

    N, M = dct_block.shape
    output_block = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            sum = 0
            for u in range(N):
                for v in range(M):
                    # Нормализация коэффициентов
                    if u == 0:
                        cu = 1 / math.sqrt(N)
                    else:
                        cu = math.sqrt(2) / math.sqrt(N)
                    if v == 0:
                        cv = 1 / math.sqrt(M)
                    else:
                        cv = math.sqrt(2) / math.sqrt(M)

                    sum += cu * cv * dct_block[u, v] * math.cos((math.pi * u * (2 * i + 1)) / (2 * N)) * math.cos(
                        (math.pi * v * (2 * j + 1)) / (2 * M))
            output_block[i, j] = sum

    return output_block


def dct_quantize_block_inverse(quantized_block: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Применяет обратное DCT к блоку, а затем квантует результат с использованием заданной матрицы квантования.

    :param quantized_block: Закодированный блок
    :param quant_matrix: Матрица квантования
    """
    # Применяем обратное квантование
    dequantized_block = quantized_block * quant_matrix
    # Применяем обратное DCT
    block = cv2.idct(dequantized_block.astype(np.float32))
    # block = idct(dequantized_block.astype(np.float32))

    return block

def reconstruct_frame_from_blocks(blocks: list, frame_shape: tuple) -> np.ndarray:
    """
    Реконструирует кадр из закодированных блоков.

    :param blocks: Список закодированных блоков.
    :param frame_shape: Размер кадра.
    :return: Реконструированный кадр.
    """
    height, width = frame_shape[:2]
    reconstructed_frame = np.zeros((height, width), dtype=np.float32)
    block_index = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if block_index < len(blocks):
                block = blocks[block_index]
                if i + 8 <= height and j + 8 <= width:
                    reconstructed_frame[i:i+8, j:j+8] = block
                else:
                    # Crop block to fit within frame dimensions
                    cropped_block = block[:min(8, height - i), :min(8, width - j)]
                    reconstructed_frame[i:i+cropped_block.shape[0], j:j+cropped_block.shape[1]] = cropped_block
                block_index += 1
    return reconstructed_frame

def process_and_decompress_frame(
        encoded_blocks: list,
        quant_matrix: np.ndarray,
        huffman_table: dict,
        frame_shape: tuple
) -> np.ndarray:
    """
    Обрабатывает и декомпрессирует кадр.

    :param encoded_blocks: Список закодированных блоков.
    :param quant_matrix: Матрица квантования.
    :param huffman_table: Таблица Хаффмана.
    :param frame_shape: Размеры исходного кадра Y (высота, ширина).
    :return: Декомпрессированный кадр.
    """
    blocks = []
    for encoded_block in encoded_blocks:
        # Декодирование Хаффмана
        decoded_data = huffman_decoding(encoded_block, huffman_table)
        # Декодирование длин серий
        rle_decoded = run_length_decode(decoded_data)
        # Обратное Zigzag-сканирование
        zigzagged_block = zigzag_scan_inverse(rle_decoded)
        # Обратное квантование и DCT
        block = dct_quantize_block_inverse(zigzagged_block, quant_matrix)
        blocks.append(block)
    # Восстановление кадра из блоков
    frame = reconstruct_frame_from_blocks(blocks, frame_shape)
    return frame

def process_and_decompress_p_frame(
        encoded_blocks: list,
        quant_matrix: np.ndarray,
        huffman_table: dict,
        frame_shape: tuple
) -> np.ndarray:
    """
    Обрабатывает и декомпрессирует кадр.

    :param encoded_blocks: Список закодированных блоков.
    :param quant_matrix: Матрица квантования.
    :param huffman_table: Таблица Хаффмана.
    :param frame_shape: Размеры исходного кадра Y (высота, ширина).
    :return: Декомпрессированный кадр.
    """
    blocks = []
    for encoded_block in encoded_blocks:
        # Декодирование Хаффмана
        decoded_data = huffman_decoding(encoded_block, huffman_table)
        # Декодирование длин серий
        rle_decoded = run_length_decode(decoded_data)
        # Обратное Zigzag-сканирование
        zigzagged_block = zigzag_scan_inverse(rle_decoded)
        # Обратное квантование и DCT
        block = dct_quantize_block_inverse(zigzagged_block, quant_matrix)
        blocks.append(block)
    # Восстановление кадра из блоков
    frame = reconstruct_frame_from_blocks(blocks, frame_shape)
    return frame

def apply_upsampling(
        u_frame: np.ndarray,
        v_frame: np.ndarray,
        frame_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет обратную субдискретизацию 4:2:0 к кадрам U и V.

    :param u_frame: Кадр U канала.
    :param v_frame: Кадр V канала.
    :param frame_shape: Размеры исходного кадра Y (высота, ширина).
    :return: Восстановленные кадры U и V каналов.
    """
    u_upsampled = cv2.resize(u_frame, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_CUBIC)
    v_upsampled = cv2.resize(v_frame, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_CUBIC)
    return u_upsampled, v_upsampled


def decode_sequence(
        encoded_frames: list,
        huffman_tables_list: list,
        frame_types: list,
        quant_matrix_Y: np.ndarray,
        quant_matrix_UV: np.ndarray,
        frame_shape: tuple
) -> list:
    """
    Декодирует последовательность кадров.

    :param encoded_frames: Список декодированных кадров.
    :param huffman_tables_list: Список таблиц Хаффмана для каждого кадра.
    :param frame_types: Список типов кадров.
    :param quant_matrix_Y: Матрица квантования для Y канала.
    :param quant_matrix_UV: Матрица квантования для U и V каналов.
    :param frame_shape: Размеры исходного кадра Y (высота, ширина).
    :return: Список декодированных кадров.
    """
    decoded_frames = []
    prev_decoded_yuv = None  # Для хранения предыдущего декодированного YUV-кадра

    for index, ((encoded_y, encoded_u, encoded_v), (huffman_table_y, huffman_table_u, huffman_table_v), frame_type) in enumerate(zip(encoded_frames, huffman_tables_list, frame_types)):
        print_progress_bar(index * 100 / len(frame_types), 100, prefix='Происходит декодирование кадров:', suffix='Готово',
                           length=50)
        # Декомпрессия разности для Y, U, V каналов
        diff_y = process_and_decompress_frame(encoded_y, quant_matrix_Y, huffman_table_y, frame_shape) if encoded_y else np.zeros(frame_shape, dtype=np.float32)
        # diff_u = process_and_decompress_frame(encoded_u, quant_matrix_UV, huffman_table_u, frame_shape) if any(encoded_u) else np.zeros((frame_shape[0]//2, frame_shape[1]//2), dtype=np.float32)
        # diff_v = process_and_decompress_frame(encoded_v, quant_matrix_UV, huffman_table_v, frame_shape) if any(encoded_v) else np.zeros((frame_shape[0]//2, frame_shape[1]//2), dtype=np.float32)
        diff_u = process_and_decompress_frame(encoded_u, quant_matrix_UV, huffman_table_u, (frame_shape[0]//2, frame_shape[1]//2)) if any(encoded_u) else np.zeros((frame_shape[0]//2, frame_shape[1]//2), dtype=np.float32)
        diff_v = process_and_decompress_frame(encoded_v, quant_matrix_UV, huffman_table_v, (frame_shape[0]//2, frame_shape[1]//2)) if any(encoded_v) else np.zeros((frame_shape[0]//2, frame_shape[1]//2), dtype=np.float32)

        # Применение апсемплинга к U и V каналам
        diff_u, diff_v = apply_upsampling(diff_u, diff_v, frame_shape)

        if frame_type == 'I':
            # I-кадр: используем декодированные данные напрямую, так как это полный кадр, а не разность
            y, u, v = diff_y, diff_u, diff_v
        else:
            # P или B кадр: добавляем разность к предыдущему кадру
            if prev_decoded_yuv is not None:
                y = cv2.add(prev_decoded_yuv[..., 0], diff_y)
                u = cv2.add(prev_decoded_yuv[..., 1], diff_u)
                v = cv2.add(prev_decoded_yuv[..., 2], diff_v)
            else:
                # Если предыдущий кадр отсутствует, используем нулевой кадр
                y, u, v = diff_y, diff_u, diff_v


        yuv_frame = cv2.merge([y.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)])
        decoded_frames.append(yuv_frame)
        prev_decoded_yuv = yuv_frame

    return decoded_frames


def predict_p_frame(prev_frame: np.ndarray, motion_vectors: np.ndarray) -> np.ndarray:
    """
    Восстанавливает P-кадр, используя оптический поток и предыдущий кадр.

    :param prev_frame: Предыдущий кадр в виде массива numpy.
    :param motion_vectors: Оптический поток в виде массива numpy.
    :return: Восстановленный P-кадр в виде массива numpy.
    """
    height, width = prev_frame.shape[:2]
    reconstructed_frame = np.zeros_like(prev_frame)

    # Проверяем, представляют ли motion_vectors смещения по обеим осям
    if motion_vectors.ndim == 3 and motion_vectors.shape[2] == 2:
        dx, dy = motion_vectors[:, :, 0], motion_vectors[:, :, 1]
    elif motion_vectors.ndim == 2:
        dx, dy = motion_vectors, np.zeros_like(motion_vectors)  # Предполагаем только горизонтальное смещение
    else:
        raise ValueError("Неправильная размерность массива motion_vectors")

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    new_x = np.clip(x + dx, 0, width - 1).astype(int)
    new_y = np.clip(y + dy, 0, height - 1).astype(int)

    reconstructed_frame[new_y, new_x] = prev_frame[y, x]

    return reconstructed_frame


def predict_b_frame(
        prev_decoded_frame: np.ndarray,
        next_decoded_frame: np.ndarray,
        forward_vectors: np.ndarray,
        backward_vectors: np.ndarray
) -> np.ndarray:
    """
    Вычисляет B-кадр, используя предыдущий и следующий декодированные кадры, а также векторы движения вперед и назад.

    :param prev_decoded_frame: Предыдущий декодированный кадр.
    :param next_decoded_frame: Следующий декодированный кадр.
    :param forward_vectors: Векторы движения вперед.
    :param backward_vectors: Векторы движения назад.
    :return: Предсказанный B-кадр.
    """
    predicted_frame = np.zeros_like(prev_decoded_frame)

    # Используем векторы движения для каждого блока для создания предсказаний из предыдущего и следующего кадров
    for block_index in range(predicted_frame.shape[0] // 8 * predicted_frame.shape[1] // 8):
        block_y, block_x = divmod(block_index, predicted_frame.shape[1] // 8)
        forward_dy, forward_dx = forward_vectors[block_index]
        backward_dy, backward_dx = backward_vectors[block_index]

        forward_block = prev_decoded_frame[block_y*8+forward_dy:block_y*8+8+forward_dy,
                                           block_x*8+forward_dx:block_x*8+8+forward_dx]
        backward_block = next_decoded_frame[block_y*8+backward_dy:block_y*8+8+backward_dy,
                                            block_x*8+backward_dx:block_x*8+8+backward_dx]

        predicted_frame[block_y*8:(block_y+1)*8, block_x*8:(block_x+1)*8] = (forward_block + backward_block) / 2

    return predicted_frame




# def unpack_from_file(file_path: str) -> tuple:
#     """
#     Загружает закодированные данные, таблицы Хаффмана, размеры кадра и типы кадров из файла.
#
#     :param file_path: Путь к файлу с закодированными данными.
#     :return: Закодированные данные, таблицы Хаффмана, размеры кадра и типы кадров.
#     """
#     # with open(file_path, "r") as file:
#     #     data_loaded = json.load(file)
#
#     with open(file_path, 'rb') as f:
#         binary_data = f.read()
#         json_string = binary_data.decode('utf-8')
#         data_loaded = json.loads(json_string)
#
#     quant_matrix_Y = np.array(data_loaded["quant_matrix_Y"])
#     quant_matrix_UV = np.array(data_loaded["quant_matrix_UV"])
#     encoded_frames = [tuple(frame) for frame in data_loaded["encoded_frames"]]  # Преобразование списков в кортежи
#
#     huffman_tables = []
#     for table_list in data_loaded["huffman_tables"]:
#         huffman_tables.append(table_list)
#
#     height = data_loaded["height"]
#     width = data_loaded["width"]
#     frame_types = data_loaded["frame_types"]
#
#     return encoded_frames, huffman_tables, height, width, frame_types, quant_matrix_Y, quant_matrix_UV

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

    # Распаковка отдельных компонентов данных
    quant_matrix_Y = np.array(data_loaded["quant_matrix_Y"])
    quant_matrix_UV = np.array(data_loaded["quant_matrix_UV"])
    encoded_frames = [tuple(frame) for frame in data_loaded["encoded_frames"]]
    huffman_tables = data_loaded["huffman_tables"]
    height = data_loaded["height"]
    width = data_loaded["width"]
    frame_types = data_loaded["frame_types"]

    return encoded_frames, huffman_tables, height, width, frame_types, quant_matrix_Y, quant_matrix_UV




def load_and_decode_sequence(file_path: str) -> list:
    """
    Загружает закодированные данные, таблицы Хаффмана, размеры кадра и типы кадров из файла
    и декодирует их.

    :param file_path: Путь к JSON-файлу
    :return: Декодированные кадры
    """
    encoded_frames, huffman_tables_list, height, width, frame_types, quant_matrix_Y, quant_matrix_UV = unpack_from_file(file_path)
    frame_shape = (height, width)

    decoded_frames = decode_sequence(encoded_frames, huffman_tables_list, frame_types, quant_matrix_Y, quant_matrix_UV,
                                     frame_shape)

    decoded_frames = convert_frames_to_rgb(decoded_frames)

    return decoded_frames


def save_decoded_frames(
        decoded_frames: list,
        save_folder: str,
        gif_filename: str = 'output_animation.gif',
        fps: int = 25
) -> None:
    """
    Сохраняет декодированные кадры в указанную папку.

    :param decoded_frames: Список декодированных кадров.
    :param save_folder: Путь к папке для сохранения кадров.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, frame in enumerate(decoded_frames):
        print_progress_bar(i + 1, len(decoded_frames), prefix='Сохраняю кадры:', suffix='Готово', length=50)
        # # Применение билатерального (двустороннего) фильтра к кадру
        # frame = cv2.bilateralFilter(frame, 8, 50, 50)
        # Применение медианного фильтра к кадру
        # blurred = median_filter(frame, 5)
        blurred = cv2.medianBlur(frame, 5)
        gaussian = cv2.GaussianBlur(blurred, (9, 9), 10.0)
        frame = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0, frame)
        filename = os.path.join(save_folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)

    # Создаем анимацию GIF из сохраненных кадров
    frame_files = sorted([f for f in os.listdir(save_folder) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(save_folder, filename)) for filename in frame_files]
    imageio.mimsave(gif_filename, images, fps=fps)
    sys.stderr.write(f"Сохраняю GIF-анимацию выходных данных с именем '{gif_filename}'\n")


def play_video(frames: list) -> None:
    """
    Воспроизводит видео.

    :param frames: Список кадров.
    :return: None
    """
    for frame in frames:
        cv2.imshow('Video Playback', frame)
        # Ждем нажатия клавиши
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Выход при нажатии 'q'
            break
    cv2.destroyAllWindows()

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет PSNR между двумя изображениями.

    :param img1: Первое изображение.
    :param img2: Второе изображение.
    :return: PSNR между двумя изображениями.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr


def calculate_quality_degradation(original_image_path: str, compressed_image_path: str, baseline_image_path: str) -> float:
    """
    Вычисляет процент деградации качества между изображениями.

    :param original_image_path: Путь к оригинальному изображению.
    :param compressed_image_path: Путь к сжатому изображению.
    :param baseline_image_path: Путь к базовому изображению.
    :return: Процент деградации качества.
    """

    # Загрузка изображений
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    compressed_image = cv2.imread(compressed_image_path, cv2.IMREAD_GRAYSCALE)
    baseline_image = cv2.imread(baseline_image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None or compressed_image is None or baseline_image is None:
        raise ValueError("Одно или несколько изображений не удалось загрузить.")

    min_size = min(original_image.shape[0], original_image.shape[1], 7)
    win_size = min_size if min_size % 2 == 1 else min_size - 1

    ssim_baseline = ssim(original_image, baseline_image, win_size=win_size)
    ssim_compressed = ssim(original_image, compressed_image, win_size=win_size)

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

    plot_metrics(psnr_list, ssim_list, degradation_list, file_names)


def plot_metrics(psnr_list: List[float], ssim_list: List[float], degradation_list: List[float], file_names: List[str]) -> None:
    """
    Построение графиков для PSNR, SSIM и процента деградации.

    :param psnr_list: Список значений PSNR.
    :param ssim_list: Список значений SSIM.
    :param degradation_list: Список значений процентной деградации.
    :param file_names: Список имен файлов.
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # График PSNR
    ax[0].plot(file_names, psnr_list, marker='o', linestyle='-', color='b')
    ax[0].set_title("PSNR over Images")
    ax[0].set_xlabel("Image")
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].grid(True)

    # График SSIM
    ax[1].plot(file_names, ssim_list, marker='o', linestyle='-', color='r')
    ax[1].set_title("SSIM over Images")
    ax[1].set_xlabel("Image")
    ax[1].set_ylabel("SSIM")
    ax[1].grid(True)

    # График процента деградации
    ax[2].plot(file_names, degradation_list, marker='o', linestyle='-', color='g')
    ax[2].set_title("Quality Degradation over Images")
    ax[2].set_xlabel("Image")
    ax[2].set_ylabel("Degradation (%)")
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.stderr.write("Начинаю процесс декодирования видео...\n")
    decoded_frames = load_and_decode_sequence('data.bin')
    # Пути к папкам с оригинальными и декодированными изображениями
    encoded_folder = 'data/encoded_frames/'
    decoded_folder = 'data/decoded_frames/'
    q90_folder = 'data/frames_Q90/'
    save_decoded_frames(decoded_frames, decoded_folder)
    sys.stderr.write(f"Начинаю вычислять PSNR, SSIM и процент деградации между оригинальными и декодированными изображениями: \n")
    # Вычисление PSNR и коэффициента деградации
    average_psnr, psnr_list, average_degradation, degradation_list, average_ssim, ssim_list = process_sequence_psnr_and_degradation(encoded_folder, decoded_folder, q90_folder)
