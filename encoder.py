import gzip
import os
import pickle
import sys
import time
import cv2
import numpy as np
from huffman import huffman_decode, huffman_encode, build_huffman_tree
from determine_frames_types import get_frame_types
from utils import print_progress_bar, save_encoded_frames, compare_folder_and_file_size, save_decoded_frames, \
    sec2timestr, rgb2yuv, quant_matrix_I, quant_matrix_PB
from length_coding import run_length_encode, run_length_decode


def get_frames(folder_path, height, width, frames_quantity):
    """
    Загружает видео из файла.

    Аргументы:
    filename -- имя файла видео.
    frames_quantity -- количество кадров для загрузки, если frames_quantity == 0, загружаются все кадры.

    Возвращает:
    frames -- данные видео в формате numpy массива.
    """

    frames = extract_frames(folder_path, height, width, frames_quantity)
    save_encoded_frames(frames, "data/encoded_frames/")

    # Преобразуем список кадров в 4D numpy массив
    frames = np.stack(frames, axis=3)

    return frames

def extract_frames(folder_path: str, height: int, width: int, frames_quantity: int) -> list:
    """
    Извлекает изображения из указанной папки.

    param folder_path: Путь к папке с изображениями
    param height: Высота изображения
    param width: Ширина изображения
    param frames_quantity: Количество изображений
    return: Список изображений
    """
    # Список для хранения кадров
    frames = []

    # Цикл по номерам кадров
    for i in range(1, frames_quantity+1):
        # Формируем имя файла RAW
        file_path = os.path.join(folder_path, f'frame_{i}.RAW')

        # Читаем RAW файл
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
            # Преобразуем данные в массив NumPy и изменяем форму массива в соответствии с разрешением и количеством каналов
            img = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            frames.append(img[:, :, ::-1])

        except FileNotFoundError:
            print(f'Файл не найден: {file_path}')
        except Exception as e:
            print(f'Ошибка при чтении файла {file_path}: {e}')

    return frames

def encode_frames(frames, scale):
    """
    Кодирует видеоролик, используя заданный шаблон типов кадров.

    Аргументы:
    frames -- входной видеоролик в формате RGB.
    scale -- масштаб качества.

    Возвращает:
    mpeg -- список с закодированными данными каждого кадра.
    """
    frame_y = []
    # Перебор кадров
    frames_quantity = frames.shape[3]
    for i in range(frames_quantity):
        # Получаем кадр
        frame = frames[:, :, :, i].astype(np.float32)
        frame = rgb2yuv(frame)
        frame_y.append(frame[:, :, 0])

    frame_types_list, frame_types = get_frame_types(frame_y)
    sys.stderr.write(f'Типы кадров: {frame_types_list} \n')

    mpeg = []
    prev_frame = None

    for i in range(frames_quantity):
        # Получаем кадр
        frame = frames[:, :, :, i].astype(np.float32)

        # Конвертация кадра в YCbCr
        frame = rgb2yuv(frame)

        # Получаем тип кадра из шаблона
        frame_type = frame_types[i % len(frame_types)]

        # Кодирование кадра
        encoded_frame, prev_frame = encode_frame(frame, frame_type, prev_frame, scale)

        # Сохраняем результат
        mpeg.append(encoded_frame)

        print_progress_bar((i+1) * 100 / frames_quantity, 100, prefix='Происходит кодирование кадров:', suffix='Готово', length=50)

    return mpeg

def encode_frame(frame, frame_type, prev_frame, scale=32):
    """
    Кодирует кадр, разбивая его на макроблоки и кодируя каждый макроблок.

    Аргументы:
    frame -- текущий кадр для кодирования.
    frame_type -- тип кадра ('I' или 'P').
    prev_frame -- предыдущий кадр.
    scale -- масштаб качества.

    Возвращает:
    mpeg -- массив со структурами данных для каждого макроблока.
    encoded_frame -- декодированный кадр после кодирования всех макроблоков.
    """
    M, N, _ = frame.shape
    block_size = (M // 16, N // 16)
    mpeg = np.empty(block_size, dtype=object)
    encoded_frame = np.zeros_like(frame)

    # Яркостная компонента предыдущего кадра
    if prev_frame is not None:
        prev_frame_y = prev_frame[:, :, 0]
    else:
        prev_frame_y = np.zeros_like(frame[:, :, 0])


    # Перебор макроблоков
    for m in range(block_size[0]):
        for n in range(block_size[1]):
            # Вычисляем координаты текущего макроблока
            x = 16 * m
            y = 16 * n
            x_range = slice(x, x + 16)
            y_range = slice(y, y + 16)

            # Кодируем макроблок
            mpeg[m, n], encoded_frame[x_range, y_range, :] = encode_block(
                frame[x_range, y_range, :], frame_type, prev_frame, prev_frame_y, x, y, scale)

    return mpeg, encoded_frame

def encode_block(block, frame_type, prev_frame, prev_frame_y, x, y, scale):
    """
    Кодирование макроблока.

    Аргументы:
    block -- текущий макроблок.
    frame_type -- тип кадра ('I' или 'P').
    prev_frame -- предыдущий кадр.
    prev_frame_y -- яркостная компонента предыдущего кадра.
    x, y -- координаты начала макроблока.
    scale -- масштаб качества.

    Возвращает:
    mpeg -- структура данных с информацией о макроблоке.
    decoded_block -- декодированный макроблок.
    """
    # Квантовые матрицы
    q1, q2 = quant_matrix_I(), quant_matrix_PB()[1]  # используем вторую матрицу из qinter

    # Инициализация структуры MPEG
    mpeg = {
        'type': 'I',
        'mvx': 0,
        'mvy': 0,
        'scale': [scale] * 6,
        'coef': np.zeros((8, 8, 6)),
        'huffman': [],
        'huffman_tree': []
    }

    # Нахождение векторов движения для P-кадров
    if frame_type == 'P':
        mpeg['type'] = 'P'
        mpeg, error_block = get_motion_vectors(mpeg, block, prev_frame, prev_frame_y, x, y)
        block = error_block  # используем блок ошибки для кодирования
        q = q2
    else:
        q = q1

    # Получение блоков яркости и цветности
    b = getblocks(block)

    # Кодирование блоков
    for i in range(6):
        coef = cv2.dct(b[:, :, i])
        # coef = dct(b[:, :, i])
        mpeg['coef'][:, :, i] = np.round(8 * coef / (scale * q))
        # Применяем RLE
        rle_encoded = run_length_encode(mpeg['coef'][:, :, i])
        # Подсчет частот для Хаффмана
        freq = {}
        for symbol, _ in rle_encoded:
            if symbol in freq:
                freq[symbol] += 1
            else:
                freq[symbol] = 1
        # Строим дерево Хаффмана и кодируем
        huffman_tree = build_huffman_tree(freq)
        encoded_data = huffman_encode(rle_encoded, huffman_tree)

        mpeg['huffman'].append(encoded_data)
        mpeg['huffman_tree'].append(huffman_tree)

    # Декодирование этого макроблока для использования в будущем P-кадре
    decoded_block = decode_block(mpeg, prev_frame, x, y)

    return mpeg, decoded_block

def decode_block(mpeg, prev_frame, x, y):
    """
    Декодирование макроблока из MPEG потока.

    Аргументы:
    mpeg -- словарь или объект с данными MPEG (тип кадра, векторы движения, коэффициенты, масштабы).
    prev_frame -- предыдущий кадр (для использования в предсказании).
    x, y -- координаты начала макроблока в предыдущем кадре.

    Возвращает:
    block -- декодированный макроблок.
    """
    # Инициализация матриц квантования
    q1, q2 = quant_matrix_I(), quant_matrix_PB()[1]  # Предположим, что qinter возвращает матрицу

    block = np.zeros((16, 16, 3))

    # Предсказание с использованием векторов движения
    if mpeg['type'] == 'P':
        block = prev_frame[x + mpeg['mvx'] : x + mpeg['mvx'] + 16, y + mpeg['mvy'] : y + mpeg['mvy'] + 16, :]
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
    block += putblocks(b)  # предполагается, что putblocks правильно собирает блоки

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

    # Четыре блока яркости
    block[0:8, 0:8, 0] = b[:,:,0]   # Верхний левый
    block[0:8, 8:16, 0] = b[:,:,1]  # Верхний правый
    block[8:16, 0:8, 0] = b[:,:,2]  # Нижний левый
    block[8:16, 8:16, 0] = b[:,:,3] # Нижний правый

    # Два подвыборочных блока цветности
    z = np.array([[1, 1], [1, 1]])
    block[:,:,1] = np.kron(b[:,:,4], z)  # Кронекер для Cb
    block[:,:,2] = np.kron(b[:,:,5], z)  # Кронекер для Cr

    return block


def getblocks(block):
    """
    Извлекает блоки из макроблока.

    Аргументы:
    block -- входной макроблок.

    Возвращает:
    b -- массив блоков 8x8 для дальнейшей обработки.
    """
    b = np.zeros((8, 8, 6))

    # Четыре блока яркости
    b[:, :, 0] = block[0:8, 0:8, 0]
    b[:, :, 1] = block[0:8, 8:16, 0]
    b[:, :, 2] = block[8:16, 0:8, 0]
    b[:, :, 3] = block[8:16, 8:16, 0]

    # Два подвыборочных блока цветности
    b[:, :, 4] = 0.25 * (block[0:16:2, 0:16:2, 1] + block[0:16:2, 1:16:2, 1] +
                         block[1:16:2, 0:16:2, 1] + block[1:16:2, 1:16:2, 1])
    b[:, :, 5] = 0.25 * (block[0:16:2, 0:16:2, 2] + block[0:16:2, 1:16:2, 2] +
                         block[1:16:2, 0:16:2, 2] + block[1:16:2, 1:16:2, 2])

    return b


def get_motion_vectors(mpeg, block, prev_frame, prev_frame_y, x, y):
    """
    Получение векторов движения для макроблока.

    Аргументы:
    mpeg -- словарь для хранения векторов движения.
    block -- текущий макроблок.
    prev_frame -- предыдущий кадр.
    prev_frame_y -- яркостная компонента предыдущего кадра.
    x, y -- координаты начала текущего макроблока.

    Возвращает:
    mpeg -- обновленный словарь с векторами движения.
    error_block -- блок ошибок.
    """
    # Работаем только с яркостной компонентой
    mby = block[:, :, 0]
    M, N = prev_frame_y.shape

    # Логарифмический поиск
    step = 8
    dx = [0, 1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 0, 1, 1, 1, 0, -1, -1, -1]

    mvx, mvy = 0, 0
    while step >= 1:
        minsad = float('inf')
        best_i = -1

        for i in range(len(dx)):
            tx = slice(x + mvx + dx[i] * step, x + mvx + dx[i] * step + 16)
            ty = slice(y + mvy + dy[i] * step, y + mvy + dy[i] * step + 16)

            if tx.start < 0 or tx.stop > M or ty.start < 0 or ty.stop > N:
                continue

            sad = np.sum(np.abs(mby - prev_frame_y[tx, ty]))

            if sad < minsad:
                minsad = sad
                best_i = i

        if best_i == -1:
            break

        mvx += dx[best_i] * step
        mvy += dy[best_i] * step

        step //= 2

    mpeg['mvx'], mpeg['mvy'] = mvx, mvy

    # Вычисляем блок ошибок
    error_block = block - prev_frame[slice(x + mvx, x + mvx + 16), slice(y + mvy, y + mvy + 16), :]

    return mpeg, error_block


if __name__ == '__main__':

    sys.stderr.write("\nРеализация кодирования MPEG2 by DB\n")

    # Параметры видео
    height, width = 240, 320
    # height, width = 1920, 1080
    # height, width = 1920, 3840
    frames_quantity = 1076

    if frames_quantity == 0:
        sys.stderr.write("\nКодируем все кадры \n")
    else:
        sys.stderr.write(f'\nКоличество кадров для загрузки: {frames_quantity} \n')

    # Loading the video
    folder_path = 'data/frames/'
    frames = get_frames(folder_path, height, width, frames_quantity)
    # Масштаб качества
    scale = 32
    sys.stderr.write("\nМасштаб качества: " + str(scale) + " \n")

    # Encoding
    start_time = time.time()
    mpeg = encode_frames(frames, scale)

    sys.stderr.write(f'\nСохраняю результат кодирования в файл data.bin \n')

    total_iterations = 2
    # Сериализация данных в байты с использованием pickle
    serialized_data = pickle.dumps(mpeg)

    print_progress_bar(1, total_iterations, prefix='Сериализация данных:', suffix='Готово', length=50)

    # Сжатие данных
    compressed_data = gzip.compress(serialized_data)
    with open('data.bin', 'wb') as file:
        file.write(compressed_data)
    print_progress_bar(2, total_iterations, prefix=' Запись данных:', suffix='Готово', length=50)

    encode_time = time.time() - start_time
    sys.stderr.write(f'\nОбщее время кодирования: {sec2timestr(encode_time)}\n')

    sys.stderr.write(f'\nКоэффициент сжатия: {compare_folder_and_file_size("data/frames", "data.bin")}\n')

