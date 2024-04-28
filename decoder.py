import ast
import gzip
import pickle
import imageio.v2 as imageio
import cv2
from huffman import huffman_decoding
import os
import numpy as np


def convert_frames_to_rgb(frames: list) -> list:
    """
    Конвертирует кадры из формата YUV в RGB.

    :param frames: Список кадров в формате YUV
    :return: Список кадров в формате RGB
    """
    # Матрица преобразования YUV в RGB
    transform_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    offset = np.array([-179.456, 135.45984, -226.816])

    rgb_frames = []
    for frame in frames:
        rgb_frame = np.dot(frame.reshape(-1, 3) + offset, transform_matrix.T)
        rgb_frame = np.clip(rgb_frame, 0, 255)
        rgb_frame = rgb_frame.reshape(frame.shape).astype(np.uint8)
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
    matrix = np.zeros((size, size), dtype=int)
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
    # Расчет новых размеров для U и V компонентов
    new_height, new_width = frame_shape

    # Увеличение размера U компонента
    upsampled_u = cv2.resize(u_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Увеличение размера V компонента
    upsampled_v = cv2.resize(v_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return upsampled_u, upsampled_v


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
        # Декомпрессия разности для Y, U, V каналов
        diff_y = process_and_decompress_frame(encoded_y, quant_matrix_Y, huffman_table_y, frame_shape) if encoded_y else np.zeros(frame_shape, dtype=np.float32)
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
                y = prev_decoded_yuv[..., 0] + diff_y
                u = prev_decoded_yuv[..., 1] + diff_u
                v = prev_decoded_yuv[..., 2] + diff_v
            else:
                # Если предыдущий кадр отсутствует, используем нулевой кадр
                y, u, v = diff_y, diff_u, diff_v



        # Объединение каналов Y, U, V в YUV кадр и его преобразование в BGR для отображения
        yuv_frame = cv2.merge([y.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)])
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        decoded_frames.append(frame)

        # Обновление ссылки на последний декодированный YUV-кадр
        prev_decoded_yuv = yuv_frame

    return decoded_frames

# def decode_sequence(encoded_frames, huffman_tables_list, frame_types, quant_matrix_Y, quant_matrix_UV, frame_shape):
#     decoded_frames = [None] * len(frame_types)  # Массив для хранения декодированных кадров
#     to_decode = set(range(len(frame_types)))  # Индексы кадров, которые еще предстоит декодировать
#
#     # Первичное декодирование I-кадров
#     for index, frame_type in enumerate(frame_types):
#         if frame_type == 'I':
#             y = process_and_decompress_frame(encoded_frames[index][0], quant_matrix_Y, huffman_tables_list[index][0], frame_shape)
#             u = process_and_decompress_frame(encoded_frames[index][1], quant_matrix_UV, huffman_tables_list[index][1], (frame_shape[0]//2, frame_shape[1]//2))
#             v = process_and_decompress_frame(encoded_frames[index][2], quant_matrix_UV, huffman_tables_list[index][2], (frame_shape[0]//2, frame_shape[1]//2))
#             u, v = apply_upsampling(u, v, frame_shape)
#             decoded_frames[index] = cv2.merge([y.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)])
#             to_decode.remove(index)  # Отмечаем кадр как декодированный
#
#     # Итеративное декодирование P и B кадров
#     something_decoded = True
#     while something_decoded:
#         something_decoded = False
#         for index, frame_type in enumerate(frame_types):
#             if index in to_decode and (frame_type == 'P' or frame_type == 'B'):
#                 prev_index = index - 1
#                 next_index = index + 1 if index + 1 < len(frame_types) else None
#
#                 can_decode = True
#                 if frame_type == 'P' and prev_index >= 0 and decoded_frames[prev_index] is None:
#                     can_decode = False
#                 if frame_type == 'B' and (prev_index < 0 or decoded_frames[prev_index] is None or next_index is None or decoded_frames[next_index] is None):
#                     can_decode = False
#
#                 if can_decode:
#                     y = process_and_decompress_frame(encoded_frames[index][0], quant_matrix_Y, huffman_tables_list[index][0], frame_shape)
#                     u = process_and_decompress_frame(encoded_frames[index][1], quant_matrix_UV, huffman_tables_list[index][1], (frame_shape[0]//2, frame_shape[1]//2))
#                     v = process_and_decompress_frame(encoded_frames[index][2], quant_matrix_UV, huffman_tables_list[index][2], (frame_shape[0]//2, frame_shape[1]//2))
#                     u, v = apply_upsampling(u, v, frame_shape)
#                     if frame_type == 'P':
#                         y += decoded_frames[prev_index][..., 0]
#                         u += decoded_frames[prev_index][..., 1]
#                         v += decoded_frames[prev_index][..., 2]
#                     elif frame_type == 'B':
#                         y = (y + decoded_frames[prev_index][..., 0] + decoded_frames[next_index][..., 0]) // 2
#                         u = (u + decoded_frames[prev_index][..., 1] + decoded_frames[next_index][..., 1]) // 2
#                         v = (v + decoded_frames[prev_index][..., 2] + decoded_frames[next_index][..., 2]) // 2
#                     decoded_frames[index] = cv2.merge([y.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)])
#                     to_decode.remove(index)
#                     something_decoded = True
#
#     # Проверка, все ли кадры декодированы
#     if any(to_decode):
#         raise Exception(f"Deadlock in frame decoding: cannot progress with available frames: {to_decode}")
#
#     return decoded_frames
#



def predict_p_frame(
        prev_decoded_frame: np.ndarray,
        current_encoded_frame: np.ndarray,
        motion_vectors: np.ndarray,
        quant_matrix: np.ndarray,
        huffman_table: dict
) -> np.ndarray:
    """
    Вычисляет P-кадр, используя предыдущий декодированный кадр и векторы движения.
    Применяет векторы движения для восстановления текущего кадра из предыдущего.
    """
    predicted_frame = np.zeros_like(prev_decoded_frame)

    # Декодируем текущий кадр для получения разности (если это необходимо)
    current_frame_diff = process_and_decompress_frame(current_encoded_frame, quant_matrix, huffman_table, prev_decoded_frame.shape)

    # Применяем векторы движения для каждого блока
    height, width = prev_decoded_frame.shape[:2]
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block_y, block_x = i // 8, j // 8
            dy, dx = motion_vectors[block_y * (width // 8) + block_x]
            src_y, src_x = max(min(i + dy, height - 8), 0), max(min(j + dx, width - 8), 0)
            predicted_frame[i:i+8, j:j+8] = prev_decoded_frame[src_y:src_y+8, src_x:src_x+8] + current_frame_diff[i:i+8, j:j+8]

    return predicted_frame


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

    # rgb_frames = convert_frames_to_rgb(decoded_frames)

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
        filename = os.path.join(save_folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)

    # Создаем анимацию GIF из сохраненных кадров
    frame_files = sorted([f for f in os.listdir(save_folder) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(save_folder, filename)) for filename in frame_files]
    imageio.mimsave(gif_filename, images, fps=fps)
    print(f"Сохраняю GIF-анимацию выходных данных с именем '{gif_filename}'")


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


if __name__ == '__main__':
    print("Начинаю процесс декодирования видео...")
    decoded_frames = load_and_decode_sequence('data.bin')
    decoded_frames_path = "data/decoded_frames/"
    print(f"Декодирование завершено. Сохраняю результаты в папку '{decoded_frames_path}'...")
    save_decoded_frames(decoded_frames, decoded_frames_path)
