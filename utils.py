import math
import os
import sys

import cv2
import imageio
import numpy as np


def unstack(movdata):
    # Проверяем, что массив имеет 4 измерения
    if movdata.ndim != 4:
        raise ValueError("Массив должен быть 4D.")

    # Получаем количество кадров
    num_frames = movdata.shape[3]

    # Разбиваем 4D массив обратно в список 3D массивов
    frames = [movdata[:, :, :, i] for i in range(num_frames)]

    return frames

def save_encoded_frames(
        encoded_frames: list,
        save_folder: str,
        gif_filename: str = 'input_animation.gif',
        fps: int = 25
) -> None:

    """
    Сохраняет декодированные кадры в указанную папку.

    :param decoded_frames: Список декодированных кадров.
    :param save_folder: Путь к папке для сохранения кадров.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    jpeg_folder = 'data/frames_Q90'
    if not os.path.exists(jpeg_folder):
        os.makedirs(jpeg_folder)

    for i, frame in enumerate(encoded_frames):
        print_progress_bar(i + 1, len(encoded_frames), prefix='Сохраняю кадры:', suffix='Готово', length=50)
        jpeg_filename = os.path.join(jpeg_folder, f"frame_{i:03d}.jpg")
        imageio.imwrite(jpeg_filename, frame[:, :, ::-1], format='JPEG', quality=90)
        filename = os.path.join(save_folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)

    # Создаем анимацию GIF из сохраненных кадров
    frame_files = sorted([f for f in os.listdir(save_folder) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(save_folder, filename)) for filename in frame_files]
    imageio.mimsave(gif_filename, images, fps=fps)
    sys.stderr.write(f"Сохраняю GIF-анимацию входных данных с именем '{gif_filename}'\n")

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

    decoded_frames = unstack(decoded_frames)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, frame in enumerate(decoded_frames):
        print_progress_bar(i + 1, len(decoded_frames), prefix='Сохраняю кадры:', suffix='Готово', length=50)
        filename = os.path.join(save_folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)

    # Создаем анимацию GIF из сохраненных кадров
    frame_files = sorted([f for f in os.listdir(save_folder) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(save_folder, filename)) for filename in frame_files]
    imageio.mimsave(gif_filename, images, fps=fps)
    sys.stderr.write(f"Сохраняю GIF-анимацию выходных данных с именем '{gif_filename}'\n")


def compare_folder_and_file_size(folder_path: str, file_path: str) -> int:
    """
    Сравнивает размер папки с размером файла.
    :param folder_path: Путь к папке
    :param file_path: Путь к файлу
    :return: Разница в размере в байтах (положительное число означает, что папка больше)
    """
    folder_size = get_folder_size(folder_path)
    file_size = os.path.getsize(file_path)

    sys.stderr.write(f"Размер папки: {folder_size} байт\n")
    sys.stderr.write(f"Размер файла '{file_path}': {file_size} байт\n")

    return folder_size / file_size

def get_folder_size(folder_path: str) -> int:
    """
    Вычисляет общий размер файлов в папке и её подпапках.

    :param folder_path: Путь к папке
    :return: Размер папки в байтах
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Суммируем размеры файлов
            total_size += os.path.getsize(file_path)
    return total_size

quant_matrix_Y = np.array([
    [2, 2, 3, 4, 5, 6, 8, 11],
    [2, 2, 2, 4, 5, 7, 9, 11],
    [3, 2, 3, 5, 7, 9, 11, 12],
    [4, 4, 5, 7, 9, 11, 12, 12],
    [5, 5, 7, 9, 11, 12, 12, 12],
    [6, 7, 9, 11, 12, 12, 12, 12],
    [8, 9, 11, 12, 12, 12, 12, 12],
    [11, 11, 12, 12, 12, 12, 12, 12]
])

quant_matrix_UV = np.array([
    [3, 3, 7, 13, 15, 15, 15, 15],
    [3, 4, 7, 13, 14, 12, 12, 12],
    [7, 7, 13, 14, 12, 12, 12, 12],
    [13, 13, 14, 12, 12, 12, 12, 12],
    [15, 14, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12],
    [15, 12, 12, 12, 12, 12, 12, 12]
])


def get_motion_vec(mpeg, mb_component, pf_component, x, y, block_index_x, block_index_y):
    """
    Определяет векторы движения в видеопотоке для заданной компоненты (Y, U, V).
    :param mpeg: массив для хранения векторов движения
    :param mb_component: компонента макроблока (Y, U или V)
    :param pf_component: соответствующая компонента предыдущего кадра
    :param x: начальная координата x макроблока
    :param y: начальная координата y макроблока
    :param block_index_x: индекс макроблока по x
    :param block_index_y: индекс макроблока по y
    """
    M, N = pf_component.shape  # Размеры компоненты предыдущего кадра

    step = 8
    dx = [0, 1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 0, 1, 1, 1, 0, -1, -1, -1]

    mvx, mvy = 0, 0
    while step >= 1:
        minsad = float('inf')
        for i in range(len(dx)):
            tx = x + mvx + dx[i] * step
            ty = y + mvy + dy[i] * step

            if tx < 1 or tx + mb_component.shape[0] > M or ty < 1 or ty + mb_component.shape[1] > N:
                continue

            sad = np.sum(np.abs(mb_component - pf_component[tx:tx + mb_component.shape[0], ty:ty + mb_component.shape[1]]))

            if sad < minsad:
                minsad = sad
                best_i = i

        mvx += dx[best_i] * step
        mvy += dy[best_i] * step
        step //= 2

    # Обновляем массив векторов движения
    mpeg[block_index_x, block_index_y, 0] = mvx
    mpeg[block_index_x, block_index_y, 1] = mvy

    # Расчёт блока ошибки
    emb = mb_component - pf_component[x + mvx:x + mvx + mb_component.shape[0], y + mvy:y + mvy + mb_component.shape[1]]

    return mpeg, emb


def apply_motion_vectors_to_frame(prev_frame, motion_vectors, frame_shape):
    """ Применяет векторы движения к предыдущему кадру для предсказания текущего кадра """
    # Преобразование списка кортежей в массив NumPy
    if isinstance(motion_vectors, list):
        try:
            motion_vectors = np.array(motion_vectors, dtype=np.int64).reshape((frame_shape[0] // 16, frame_shape[1] // 16, 2))
        except OverflowError as e:
            print(f"Error converting motion vectors: {e}")
            raise

    predicted_frame = np.zeros_like(prev_frame)
    for i in range(0, frame_shape[0], 16):
        for j in range(0, frame_shape[1], 16):
            block_index_x = i // 16
            block_index_y = j // 16
            dx, dy = motion_vectors[block_index_x, block_index_y]
            predicted_frame[i:i + 16, j:j + 16] = np.roll(np.roll(prev_frame, dy, axis=1), dx, axis=0)[i:i + 16,
                                                  j:j + 16]
    return predicted_frame

def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = '',
        suffix: str = '',
        decimals: int = 1,
        length: int = 100,
        fill: str = '█'
):
    """
    Выводит прогресс-бар с указанными параметрами.
    params:
        iteration   - Required  : Текущая итерация
        total       - Required  : Всего итераций
        prefix      - Optional  : Строка префикса
        suffix      - Optional  : Строка суффикса
        decimals    - Optional  : Положительное число знаков после запятой
        length      - Optional  : Длина прогресс-бара
        fill        - Optional  : Символ заполнения прогресс-бара

    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stderr.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stderr.flush()
    # Print New Line on Complete
    if iteration == total:
        sys.stderr.write('\n')
        sys.stderr.flush()

def sec2timestr(sec):
    """ Конвертирует секунды в удобочитаемый формат времени. """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f'{h:02}:{m:02}:{s:05.2f}'

def play_video(frames):
    for frame in frames:
        cv2.imshow('Video Playback', frame)
        # Ждем нажатия клавиши
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Выход при нажатии 'q'
            break
    cv2.destroyAllWindows()

def dct(block: np.ndarray) -> np.ndarray:
    """
    Вычисляет двумерное дискретное косинусное преобразование (DCT) для данного блока.
    DCT преобразует блок из пространственной области в частотную.

    param block: Входящий блок
    return: Выходной блок

    """

    N, M = block.shape
    dct_block = np.zeros((N, M))

    for u in range(N):
        for v in range(M):
            sum = 0
            for i in range(N):
                for j in range(M):
                    sum += block[i, j] * math.cos((math.pi * u * (2 * i + 1)) / (2 * N)) * math.cos(
                        (math.pi * v * (2 * j + 1)) / (2 * M))
            # Нормализация коэффициентов
            if u == 0:
                cu = 1 / math.sqrt(N)
            else:
                cu = math.sqrt(2) / math.sqrt(N)
            if v == 0:
                cv = 1 / math.sqrt(M)
            else:
                cv = math.sqrt(2) / math.sqrt(M)
            dct_block[u, v] = cu * cv * sum

    return dct_block


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

def rgb2yuv(rgb):
    """
    Конвертация из RGB в YCbCr.

    Аргументы:
    rgb -- входное изображение в RGB формате.

    Возвращает:
    yuv -- изображение в YCbCr формате.
    """
    # Матрица преобразования из RGB в YCbCr
    m = np.array([[ 0.299,     0.587,     0.144],
                  [-0.168736, -0.331264,  0.5],
                  [ 0.5,      -0.418688, -0.081312]])

    # Получаем размеры изображения
    nr, nc, c = rgb.shape

    # Переформатирование массива для матричного умножения
    rgb = rgb.reshape(nr*nc, 3)

    # Преобразование цветового кодирования
    yuv = np.dot(m, rgb.T)
    yuv = yuv + np.array([[0], [0.5], [0.5]])  # Добавляем смещение к Cb и Cr

    # Возвращаем к исходному размеру
    yuv = yuv.T.reshape(nr, nc, c)
    return yuv

def yuv2rgb(yuv):
    """
    Конвертация из YCbCr в RGB.

    Аргументы:
    yuv -- входное изображение в YCbCr формате.

    Возвращает:
    rgb -- изображение в RGB формате.
    """
    # Матрица преобразования из YCbCr в RGB
    m = np.array([[ 0.299,     0.587,     0.144],
                  [-0.168736, -0.331264,  0.5],
                  [ 0.5,      -0.418688, -0.081312]])
    m = np.linalg.inv(m)  # Вычисляем обратную матрицу

    # Получаем размеры изображения
    nr, nc, c = yuv.shape

    # Переформатирование массива для матричного умножения
    yuv = yuv.reshape(nr*nc, 3)

    # Преобразование цветового кодирования с коррекцией смещения
    rgb = yuv - np.array([0, 0.5, 0.5])
    rgb = np.dot(rgb, m.T)  # Применяем матрицу преобразования

    # Возвращаем к исходному размеру
    rgb = rgb.reshape(nr, nc, c)
    return rgb

def quant_matrix_PB():
    """
    Таблица квантования для P- или B-кадров.

    Возвращает:
    q -- таблица квантования в виде единичного значения или numpy массива 8x8, заполненного значением 16.
    """
    # Если нужно возвращать только число:
    q_value = 16
    # Если нужно возвращать матрицу:
    q_matrix = np.full((8, 8), 16)
    return q_value, q_matrix

def quant_matrix_I():
    """
    Таблица квантования для I-кадров.

    Возвращает:
    q -- таблица квантования в виде numpy массива.
    """
    q = np.array([
        [2, 2, 3, 4, 5, 6, 8, 11],
        [2, 2, 2, 4, 5, 7, 9, 11],
        [3, 2, 3, 5, 7, 9, 11, 12],
        [4, 4, 5, 7, 9, 11, 12, 12],
        [5, 5, 7, 9, 11, 12, 12, 12],
        [6, 7, 9, 11, 12, 12, 12, 12],
        [8, 9, 11, 12, 12, 12, 12, 12],
        [11, 11, 12, 12, 12, 12, 12, 12]
    ])
    return q