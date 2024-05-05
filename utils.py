import sys

import numpy as np

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

zigzag_order = [
    (0, 0),
    (0, 1), (1, 0),
    (2, 0), (1, 1), (0, 2),
    (0, 3), (1, 2), (2, 1), (3, 0),
    (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
    (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
    (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
    (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
    (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
    (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
    (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
    (4, 7), (5, 6), (6, 5), (7, 4),
    (7, 5), (6, 6), (5, 7),
    (6, 7), (7, 6),
    (7, 7)
]
zigzag_index = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]


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