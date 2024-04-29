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