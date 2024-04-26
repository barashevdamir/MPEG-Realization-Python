import imageio
import numpy as np
import cv2
import os
from determine_frames_types import get_frame_types, calculate_optical_flow_farneback
from huffman import huffman_encoding
import json
import math


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


def extract_frames(folder_path: str, height: int, width: int) -> list:
    """
    Извлекает изображения из указанной папки.

    param folder_path: Путь к папке с изображениями
    param height: Высота изображения
    param width: Ширина изображения
    return: Список изображений
    """
    # Список для хранения кадров
    frames = []

    # Цикл по номерам кадров
    for i in range(21, 30):
        # Формируем имя файла RAW
        file_path = os.path.join(folder_path, f'frame_{i}.RAW')
        file_size = os.path.getsize(file_path)

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

def convert_frames_to_yuv(frames: list) -> list:
    """
    Конвертирует кадры в формат YUV.

    :param frames: Список кадров в формате RGB
    :return: Список кадров в формате YUV
    """
    transform_matrix = np.array([
        [0.229, 0.587, 0.144],
        [0.5, -0.4187, -0.0813],
        [0.1687, -0.3313, 0.5]
    ])
    offset = np.array([0, 128, 128])
    yuv_frames = []
    for frame in frames:

        yuv_frame = np.dot(frame.reshape(-1, 3), transform_matrix.T) + offset
        yuv_frame = np.clip(yuv_frame, 0, 255)
        yuv_frame = yuv_frame.reshape(frame.shape).astype(np.float32)
        yuv_frames.append(yuv_frame)

    return yuv_frames


def apply_subsampling(frames: list) -> list:
    """
    Применяет субдискретизацию 4:2:0 ко всем кадрам в списке.

    :param frames: Список кадров в формате YUV
    :return: Список кадров после субдискретизации с уменьшенными U и V каналами
    """
    subsampled_frames = []

    for yuv_frame in frames:
        # Разделяем каналы
        y, u, v = cv2.split(yuv_frame)

        # Применяем субдискретизацию к каналам U и V
        u_subsampled = u[::2, ::2]
        v_subsampled = v[::2, ::2]

        # Сохраняем каналы как отдельные элементы списка
        subsampled_frames.append((y, u_subsampled, v_subsampled))

    return subsampled_frames

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

def dct_quantize_block(block: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Применяет DCT к блоку, а затем квантует результат с использованием заданной матрицы квантования.

    :param block: Входящий блок
    :param quant_matrix: Матрица квантования
    """
    # Применяем DCT
    # dct_block = cv2.dct(block.astype(np.float32))
    dct_block = dct(block.astype(np.float32))

    # Применяем квантование
    quantized_block = np.round(dct_block / quant_matrix)
    return quantized_block

def zigzag_scan(matrix: np.ndarray) -> list:
    """
    Преобразует матрицу в одномерный массив с помощью Zigzag-сканирования.

    Zigzag-порядок обеспечивает, что коэффициенты,
    которые вероятно будут иметь меньшие значения или нули (особенно после квантования),
    группируются в конце массива, что улучшает эффективность последующего сжатия данных.

    :param matrix: Входная матрица (обычно 8x8 после DCT и квантования)
    :return: Одномерный массив элементов, полученных Zigzag-сканированием
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    size = matrix.shape[0]
    result = []

    for d in range(2 * size - 1):
        for i in range(d + 1):
            j = d - i

            if i >= size or j >= size:
                continue  # Пропускаем индексы за пределами матрицы

            if d % 2 == 0:
                result.append(matrix[j, i] if d & 1 else matrix[i, j])
            else:
                result.append(matrix[i, j] if d & 1 else matrix[j, i])

    return result

def run_length_encode(arr: list) -> list:
    """
    Применяет кодирование по длинам серий к 1D массиву.

    :param arr: 1D массив
    :return: Список элементов, полученных RLE-кодированием
    """
    if not arr:
        return []
    else:
        result = [(arr[0], 1)]  # Значение, счетчик
        for element in arr[1:]:
            if element == result[-1][0]:
                result[-1] = (result[-1][0], result[-1][1] + 1)
            else:
                result.append((element, 1))
        return result


def process_and_compress_frame(frame: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Применяет DCT и квантование к каждому блоку 8x8 в кадре,
    затем применяет Zigzag - сканирование и RLE для сжатия.

    :param frame: Входящий кадр
    :param quant_matrix: Матрица квантования
    :return: Сжатый кадр
    """
    h, w = frame.shape
    processed_frame = np.zeros_like(frame, dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = frame[i:i+8, j:j+8]
            if block.shape == (8, 8):
                processed_block = dct_quantize_block(block, quant_matrix)
                processed_frame[i:i+8, j:j+8] = processed_block
    return processed_frame

def encode_i_frame(frame: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Кодирует I-кадр в виде последовательности RLE блоков.
    Кодирование производится с использованием таблицы Хаффмана.

    :param frame: Входящий кадр
    :param quant_matrix: Матрица квантования
    :return: Cжатый кадр
    """
    height, width = frame.shape[:2]
    all_rle_blocks = []  # Список для хранения RLE закодированных блоков
    all_data_for_huffman = []  # Список для сбора данных всех блоков для Хаффмана

    # Обработка и сбор данных из каждого блока
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = frame[i:i + 8, j:j + 8]
            padded_block = pad_block_to_8x8(block)
            quantized_block = dct_quantize_block(padded_block, quant_matrix)
            zigzagged_block = zigzag_scan(quantized_block)
            rle_block = run_length_encode(zigzagged_block)
            all_rle_blocks.append(rle_block)
            all_data_for_huffman.extend(rle_block)

    # Создание общей таблицы Хаффмана для всех блоков кадра
    huffman_table = huffman_encoding(all_data_for_huffman)
    # Кодирование блоков с использованием общей таблицы Хаффмана
    encoded_blocks = []
    for rle_block in all_rle_blocks:
        encoded_block = ''
        for value, count in rle_block:
            key = (value, count)
            code = huffman_table.get(key, 'ERROR')  # Безопасно извлекаем код
            if code == 'ERROR':
                print("ERROR")
            encoded_block += code
        encoded_blocks.append(encoded_block)

    return encoded_blocks, huffman_table

def predict_p_frame_from_flow(prev_frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Предсказывает P-кадр, используя вектор движения.

    :param prev_frame: Предыдущий кадр
    :param flow: Вектор движения
    :return: Предсказанный кадр
    """

    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))

    # Применение векторов движения к координатам пикселей
    new_positions = flow_map + flow.reshape(-1, 2)

    # Ограничение координат, чтобы они оставались в пределах изображения
    new_positions[:, 0] = np.clip(new_positions[:, 0], 0, h - 1)
    new_positions[:, 1] = np.clip(new_positions[:, 1], 0, w - 1)

    # Создание предсказанного P-кадра
    predicted_frame = np.zeros_like(prev_frame)
    for i in range(len(flow_map)):
        src_y, src_x = flow_map[i].astype(int)
        dst_y, dst_x = new_positions[i].astype(int)
        predicted_frame[dst_y, dst_x] = prev_frame[src_y, src_x]

    return predicted_frame


def predict_b_frame(prev_frame: np.ndarray, next_frame: np.ndarray, flow_to_prev: np.ndarray, flow_to_next: np.ndarray) -> np.ndarray:
    """
    Создаёт предсказание для B-кадра на основе векторов движения к предыдущему и последующему кадрам.

    :param prev_frame: Предыдущий кадр
    :param next_frame: Последующий кадр
    :param flow_to_prev: Вектор движения к предыдущему кадру
    :param flow_to_next: Вектор движения к последующему кадру
    :return: Предсказанный кадр
    """
    h, w = prev_frame.shape[:2]

    # Предсказание на основе предыдущего кадра
    predicted_from_prev = predict_p_frame_from_flow(prev_frame, flow_to_prev)

    # Предсказание на основе последующего кадра
    predicted_from_next = predict_p_frame_from_flow(next_frame, -flow_to_next)

    # Комбинирование предсказаний
    predicted_frame = (predicted_from_prev + predicted_from_next) / 2

    return predicted_frame


def encode_p_frame(prev_frame: np.ndarray, current_frame: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Кодирует P-кадр, используя предыдущий кадр и матрицу квантования.

    :param prev_frame: Предыдущий кадр
    :param current_frame: Текущий кадр
    :param quant_matrix: Матрица квантования
    :return: Кодированный кадр
    """
    # Вычисление оптического потока и создание предсказанного P-кадра
    flow = calculate_optical_flow_farneback(prev_frame, current_frame)
    predicted_frame = predict_p_frame_from_flow(prev_frame, flow)

    # Вычисление разности между текущим кадром и предсказанным
    frame_diff = cv2.absdiff(current_frame, predicted_frame)

    height, width = frame_diff.shape
    all_rle_blocks = []
    all_data_for_huffman = []

    # Разбиение frame_diff на блоки 8x8 и их обработка
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = frame_diff[i:i + 8, j:j + 8]
            padded_block = pad_block_to_8x8(block)
            quantized_block = dct_quantize_block(padded_block, quant_matrix)
            zigzagged_block = zigzag_scan(quantized_block)
            rle_block = run_length_encode(zigzagged_block)
            all_rle_blocks.append(rle_block)
            all_data_for_huffman.extend(rle_block)

    # Создание общей таблицы Хаффмана для всех блоков кадра
    huffman_table = huffman_encoding(all_data_for_huffman)
    # Кодирование блоков с использованием общей таблицы Хаффмана
    encoded_blocks = []
    for rle_block in all_rle_blocks:
        encoded_block = ''
        for value, count in rle_block:
            key = (value, count)
            code = huffman_table.get(key, 'ERROR')  # Безопасно извлекаем код
            if code == 'ERROR':
                print("ERROR")
            encoded_block += code
        encoded_blocks.append(encoded_block)

    return encoded_blocks, huffman_table

def encode_b_frame(prev_frame: np.ndarray, current_frame: np.ndarray, next_frame: np.ndarray, quant_matrix: np.ndarray) -> np.ndarray:
    """
    Кодирует B-кадр, используя предыдущий и последующий кадры.

    :param prev_frame: Предыдущий кадр
    :param current_frame: Текущий кадр
    :param next_frame: Последующий кадр
    :param quant_matrix: Матрица квантования
    :return: Кодированный кадр
    """
    # Вычисление оптического потока к предыдущему и последующему кадрам
    flow_to_prev = calculate_optical_flow_farneback(current_frame, prev_frame)
    flow_to_next = calculate_optical_flow_farneback(current_frame, next_frame)

    # Создание предсказанного B-кадра
    predicted_frame = predict_b_frame(prev_frame, next_frame, flow_to_prev, flow_to_next)

    # Вычисление разности между текущим кадром и предсказанным
    if len(current_frame.shape) == 3:  # Проверяем, есть ли 3 канала (цветное изображение)
        frame_diff = cv2.absdiff(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(predicted_frame, cv2.COLOR_BGR2GRAY))
    else:
        frame_diff = cv2.absdiff(current_frame, predicted_frame)  # Изображение уже в градациях серого

    # Кодирование разности кадров с использованием DCT и квантования
    encoded_frame = encode_p_frame(predicted_frame, frame_diff,
                                            quant_matrix)  # Используем ту же функцию, что и для P-кадров

    return encoded_frame

def encode_frame_by_type(
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        frame_type: str,
        quant_matrix_Y: np.ndarray,
        quant_matrix_UV: np.ndarray,
        prev_frames: tuple = None,
        next_frame: np.ndarray = None
) -> tuple:
    if frame_type == 'I':
        encoded_y, huffman_table_y = encode_i_frame(y, quant_matrix_Y)
        encoded_u, huffman_table_u = encode_i_frame(u, quant_matrix_UV)
        encoded_v, huffman_table_v = encode_i_frame(v, quant_matrix_UV)
        huffman_tables = (huffman_table_y, huffman_table_u, huffman_table_v)
    elif frame_type == 'P':
        assert prev_frames is not None, "Previous frame is required for P frame encoding"
        prev_y, prev_u, prev_v = prev_frames
        encoded_y, huffman_table_y = encode_p_frame(prev_y, y, quant_matrix_Y)
        encoded_u, huffman_table_u = encode_p_frame(prev_u, u, quant_matrix_UV)
        encoded_v, huffman_table_v = encode_p_frame(prev_v, v, quant_matrix_UV)
        huffman_tables = (huffman_table_y, huffman_table_u, huffman_table_v)
    elif frame_type == 'B':
        assert prev_frames is not None and next_frame is not None, "Previous and next frames are required for B frame encoding"
        prev_y, prev_u, prev_v = prev_frames
        next_y, next_u, next_v = next_frame
        encoded_y, huffman_table_y = encode_b_frame(prev_y, next_y, y, quant_matrix_Y)
        encoded_u, huffman_table_u = encode_b_frame(prev_u, next_u, u, quant_matrix_UV)
        encoded_v, huffman_table_v = encode_b_frame(prev_v, next_v, v, quant_matrix_UV)
        huffman_tables = (huffman_table_y, huffman_table_u, huffman_table_v)
    else:
        raise ValueError("Unsupported frame type")

    return (encoded_y, encoded_u, encoded_v), huffman_tables

def encode_sequence(frames: list, frame_types: list, quant_matrix_Y: np.ndarray, quant_matrix_UV: np.ndarray) -> tuple:
    """
    Закодировать последовательность кадров.

    :param frames: Список кадров
    :param frame_types: Список типов кадров
    :param quant_matrix_Y: Матрица квантования для Y-канала
    :param quant_matrix_UV: Матрица квантования для UV-канала
    :return: Список закодированных кадров и таблицы Хаффмана
    """
    encoded_frames = []
    huffman_tables_list = []
    for i, ((y, u, v), frame_type) in enumerate(zip(frames, frame_types)):
        print("Происходит кодирование кадров: ", i*100/len(frames), " %")
        prev_frames = (frames[i - 1][0], frames[i - 1][1], frames[i - 1][2]) if i > 0 else None
        next_frame = (frames[i + 1][0], frames[i + 1][1], frames[i + 1][2]) if i < len(frames) - 1 else None

        (encoded_y, encoded_u, encoded_v), huffman_tables = encode_frame_by_type(y, u, v, frame_type, quant_matrix_Y, quant_matrix_UV, prev_frames, next_frame)

        encoded_frames.append((encoded_y, encoded_u, encoded_v))
        huffman_tables_list.append(huffman_tables)

    print("Обрабатываются закодированные кадры...")
    return encoded_frames, huffman_tables_list

def prepare_data_for_json(data: dict | list | tuple) -> dict | list | tuple:
    """
    Рекурсивно преобразует ключи в словарях из кортежей в строки для сериализации JSON.

    :param data: Словарь, список или кортеж
    :return: Словарь, список или кортеж
    """
    if isinstance(data, dict):
        # Обработка словаря: преобразование всех ключей в строки
        return {str(key): prepare_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Обработка списка: рекурсивная обработка каждого элемента
        return [prepare_data_for_json(item) for item in data]
    elif isinstance(data, tuple):
        # Обработка кортежа: рекурсивная обработка каждого элемента кортежа
        return tuple(prepare_data_for_json(item) for item in data)
    else:
        return data


def pack_to_file(
        encoded_frames: list,
        huffman_tables_list: list,
        file_path: str,
        height: int,
        width: int,
        frame_types: list,
        quant_matrix_Y: np.ndarray,
        quant_matrix_UV: np.ndarray
) -> None:
    """
    Запаковать закодированные данные и таблицы Хаффмана в JSON-файл.

    :param encoded_frames: Закодированные кадры
    :param huffman_tables_list: Список таблиц Хаффмана
    :param file_path: Путь к JSON-файлу
    :param height: Высота изображения
    :param width: Ширина изображения
    :param frame_types: Список типов кадров
    :param quant_matrix_Y: Матрица квантования для Y-канала
    :param quant_matrix_UV: Матрица квантования для UV-канала
    :return: None
    """
    quant_matrix_Y_list = quant_matrix_Y.tolist() if isinstance(quant_matrix_Y, np.ndarray) else quant_matrix_Y
    quant_matrix_UV_list = quant_matrix_UV.tolist() if isinstance(quant_matrix_UV, np.ndarray) else quant_matrix_UV

    # Подготовка данных Huffman и других данных для сериализации
    prepared_data = prepare_data_for_json({
        "height": height,
        "width": width,
        "huffman_tables": huffman_tables_list,
        "encoded_frames": encoded_frames,
        "frame_types": frame_types,
        "quant_matrix_Y": quant_matrix_Y_list,
        "quant_matrix_UV": quant_matrix_UV_list,
    })
    # Запись данных в бинарный файл
    with open('data.bin', 'wb') as f:
        dumped_json_string = json.dumps(prepared_data)
        binary_string = dumped_json_string.encode('utf-8')
        f.write(binary_string)
    #
    # # Сериализация подготовленных данных в JSON
    # with open(file_path, "w") as file:
    #     json.dump(prepared_data, file)

def write_json_to_binary_file(json_file_path: str, binary_file_path: str) -> None:
    """
    Записывает JSON-файл в бинарный файл.

    :param json_file_path: Путь к JSON-файлу
    :param binary_file_path: Путь к бинарному файлу
    :return: None
    """
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    with open(binary_file_path, 'wb') as binary_file:
        json.dump(json_data, binary_file)


def pack_encoded_data_to_mdat_content(encoded_frames: list, huffman_tables_list: list) -> bytearray:
    """
    Объединяет закодированные данные и таблицы Хаффмана в один байтовый поток для 'mdat'.

    :param encoded_frames: Закодированные кадры
    :param huffman_tables_list: Список таблиц Хаффмана
    :return: Байтовый поток для 'mdat'
    """
    mdat_content = bytearray()

    # Обработка закодированных кадров
    for frame in encoded_frames:
        for channel in frame:  # channel представляет собой список строк с бинарными данными
            for binary_str in channel:
                if binary_str:
                    # Преобразование бинарной строки в байты и добавление в mdat_content
                    num_bytes = (len(binary_str) + 7) // 8
                    byte_data = int(binary_str, 2).to_bytes(num_bytes, 'big')
                    mdat_content.extend(byte_data)

    # Обработка таблиц Хаффмана
    for table in huffman_tables_list:
        for value_tuple in table:  # value_tuple - это словарь вида (значение, код)
            for value, code in value_tuple.items():  # Итерация по парам ключ-значение
                if value:  # Проверка на непустое значение
                    value_bytes = float(value).hex().encode('utf-8')  # Предполагаем, что значение - это float
                    mdat_content.extend(value_bytes)
                if code:  # Проверка на непустой код
                    code_bytes = code.encode('utf-8')
                    mdat_content.extend(code_bytes)

    return mdat_content

def play_video(frames):
    for frame in frames:
        cv2.imshow('Video Playback', frame)
        # Ждем нажатия клавиши
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Выход при нажатии 'q'
            break
    cv2.destroyAllWindows()

def save_encoded_frames(encoded_frames, save_folder, gif_filename='input_animation.gif', fps=25):
    """
    Сохраняет декодированные кадры в указанную папку.

    :param decoded_frames: Список декодированных кадров.
    :param save_folder: Путь к папке для сохранения кадров.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, frame in enumerate(encoded_frames):
        filename = os.path.join(save_folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    # Создаем анимацию GIF из сохраненных кадров
    frame_files = sorted([f for f in os.listdir(save_folder) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(save_folder, filename)) for filename in frame_files]
    imageio.mimsave(gif_filename, images, fps=fps)
    print(f"GIF saved as {gif_filename}")


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

def compare_folder_and_file_size(folder_path: str, file_path: str) -> int:
    """
    Сравнивает размер папки с размером файла.
    :param folder_path: Путь к папке
    :param file_path: Путь к файлу
    :return: Разница в размере в байтах (положительное число означает, что папка больше)
    """
    folder_size = get_folder_size(folder_path)
    file_size = os.path.getsize(file_path)

    print(f"Размер папки: {folder_size} байт")
    print(f"Размер файла '{file_path}': {file_size} байт")

    return folder_size / file_size

if __name__ == '__main__':

    folder_path = 'data/frames/'
    height, width = 1920, 1080
    frames = extract_frames(folder_path, height, width)
    frames_yuv = convert_frames_to_yuv(frames)

    save_encoded_frames(frames, "data/encoded_frames/")

    subsampled_frames = apply_subsampling(frames_yuv)

    # Создаем список, содержащий только каналы Y из субдискретизированных кадров
    y_frames = [frame[0] for frame in subsampled_frames]

    frame_types = get_frame_types(y_frames)

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

    encoded_frames, huffman_tables_list = encode_sequence(subsampled_frames, frame_types, quant_matrix_Y,
                                                          quant_matrix_UV)

    pack_to_file(
        encoded_frames,
        huffman_tables_list,
        "output_file.json",
        height,
        width,
        frame_types,
        quant_matrix_Y,
        quant_matrix_UV,
    )

    print(compare_folder_and_file_size("data/frames", "data.bin"))