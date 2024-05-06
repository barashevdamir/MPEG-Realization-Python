import numpy as np
import cv2
import os


def calculate_sad(frame1: np.ndarray, frame2: np.ndarray) -> int:
    """
    Вычисляет сумму абсолютных разностей между Y-компонентами двух кадров YUV.

    param frame1: Первый кадр.
    param frame2: Второй кадр.
    return: Сумма абсолютных разностей.
    """
    return np.sum(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))


def calculate_optical_flow_farneback(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Вычисляет оптический поток Фарнебека между Y-компонентами двух кадров YUV.

    :param prev_frame: Предыдущий кадр.
    :param next_frame: Следующий кадр.
    :return: Оптический поток Фарнебека.
    """
    opts_cv = dict(
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    # Проверяем, являются ли кадры одноканальными
    if len(prev_frame.shape) == 2 and len(next_frame.shape) == 2:
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, **opts_cv)
    else:
        raise ValueError("Кадры должны быть в формате градаций серого (одноканальные).")
    return flow

def calculate_optical_flow_lucas_canade(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Вычисляет оптический поток между двумя изображениями с использованием метода Лукаса-Канаде.

    Параметры:
    - prev_frame: первое изображение в градациях серого.
    - next_frame: второе изображение в градациях серого.

    Возвращает:
    - points: массив точек на первом изображении, где оптический поток был вычислен.
    - flow: массив векторов оптического потока для соответствующих точек.
    """
    # Параметры для алгоритма Лукаса-Канаде
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )
    # Определение точек для отслеживания
    p0 = cv2.goodFeaturesToTrack(prev_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is not None:
        # Вычисление оптического потока
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, p0, None, **lk_params)
        # Выбор хороших точек
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_old, good_new
    else:
        return None, None


def calculate_adaptive_thresholds(frames: list) -> tuple:
    """
    Вычисляет адаптивные пороги для SAD и оптического потока.

    :param frames: Список кадров.
    :return: Адаптивные пороги для SAD и оптического потока.
    """
    sad_values = []
    flow_magnitudes = []

    # Проходим через все кадры для вычисления SAD и оптического потока
    for i in range(1, len(frames) - 1):
        prev_frame = frames[i - 1]
        current_frame = frames[i]
        next_frame = frames[i + 1]

        # Вычисляем SAD
        sad_prev = calculate_sad(prev_frame, current_frame)
        sad_next = calculate_sad(current_frame, next_frame)
        sad_values.append(sad_prev)
        sad_values.append(sad_next)

        # Вычисляем оптический поток
        flow_prev = calculate_optical_flow_farneback(prev_frame, current_frame)
        flow_next = calculate_optical_flow_farneback(current_frame, next_frame)
        flow_magnitude_prev = np.sqrt(flow_prev[..., 0] ** 2 + flow_prev[..., 1] ** 2)
        flow_magnitude_next = np.sqrt(flow_next[..., 0] ** 2 + flow_next[..., 1] ** 2)
        flow_magnitudes.append(np.mean(flow_magnitude_prev))
        flow_magnitudes.append(np.mean(flow_magnitude_next))

    # Вычисляем средние значения и стандартные отклонения
    mean_sad = np.mean(sad_values)
    std_sad = np.std(sad_values)
    mean_flow = np.mean(flow_magnitudes)
    std_flow = np.std(flow_magnitudes)

    # Адаптируем порог для определения B-кадров
    threshold_b_frame_adaptive = mean_flow + std_flow / 2  # Это просто пример, может быть настроено экспериментально

    return mean_sad + std_sad, mean_flow + std_flow, threshold_b_frame_adaptive

def determine_frame_type(
        prev_frame: np.ndarray,
        current_frame: np.ndarray,
        next_frame: np.ndarray,
        threshold_sad: int = 50000,
        threshold_flow: float = 2.0,
        threshold_b_frame: float = 1.5) -> str:
    """
    Определяет тип кадра, используя SAD (Sum of absolute differences) и оптический поток Фарнебека, включая определение B-кадров.

    :param prev_frame: Предыдущий кадр.
    :param current_frame: Текущий кадр.
    :param next_frame: Следующий кадр.
    :param threshold_sad: Порог для SAD.
    :param threshold_flow: Порог для оптического потока.
    :param threshold_b_frame: Порог для определения B-кадров.
    :return: Тип кадра.
    """
    sad_prev = calculate_sad(prev_frame, current_frame)
    sad_next = calculate_sad(current_frame, next_frame)
    flow_prev = calculate_optical_flow_farneback(prev_frame, current_frame)
    flow_next = calculate_optical_flow_farneback(current_frame, next_frame)

    flow_magnitude_prev = np.sqrt(flow_prev[..., 0] ** 2 + flow_prev[..., 1] ** 2)
    avg_flow_magnitude_prev = np.mean(flow_magnitude_prev)

    flow_magnitude_next = np.sqrt(flow_next[..., 0] ** 2 + flow_next[..., 1] ** 2)
    avg_flow_magnitude_next = np.mean(flow_magnitude_next)

    # Если текущий кадр может быть хорошо предсказан и из предыдущего, и из следующего кадра, считаем его B-кадром
    if avg_flow_magnitude_prev <= threshold_flow and avg_flow_magnitude_next <= threshold_flow and sad_prev < threshold_sad and sad_next < threshold_sad:
        # return 'B'
        return 'P'
    # Если текущий кадр хорошо предсказывается из предыдущего кадра, считаем его P-кадром
    elif sad_prev < threshold_sad or avg_flow_magnitude_prev < threshold_b_frame:
        return 'P'
    # Во всех остальных случаях считаем кадр I-кадром
    else:
        return 'I'


def determine_last_frame_type(prev_frame: np.ndarray, last_frame: np.ndarray, threshold_sad: int, threshold_flow: float) -> str:
    """
    Определяет тип последнего кадра на основе его отличия от предпоследнего.

    :param prev_frame: Предыдущий кадр.
    :param last_frame: Последний кадр.
    :param threshold_sad: Порог для SAD.
    :param threshold_flow: Порог для оптического потока.
    :return: Тип последнего кадра.
    """
    sad_last = calculate_sad(prev_frame, last_frame)
    flow_last = calculate_optical_flow_farneback(prev_frame, last_frame)

    flow_magnitude_last = np.sqrt(flow_last[..., 0] ** 2 + flow_last[..., 1] ** 2)
    avg_flow_magnitude_last = np.mean(flow_magnitude_last)

    if sad_last > threshold_sad or avg_flow_magnitude_last > threshold_flow:
        return 'I'
    else:
        return 'P'

def get_frame_types(frames: list) -> list:
    """
    Определяет типы кадров.

    :param frames: Список кадров.
    :return: Список типов кадров.
    """
    if not frames:
        return []

    frame_types = ['I']
    adaptive_threshold_sad, adaptive_threshold_flow, adaptive_threshold_b_frame = calculate_adaptive_thresholds(frames)

    # Первый кадр уже помечен как I-кадр
    # for i in range(1, len(frames)):
    #     frame_types.append('I')

    # Первый кадр уже помечен как I-кадр
    for i in range(1, len(frames) - 1):
        if i % 10 == 0:
            # Каждый 10-й кадр устанавливается как I-кадр
            frame_types.append('I')
        else:
            # Для остальных кадров определяем тип
            prev_frame = frames[i - 1]
            current_frame = frames[i]
            next_frame = frames[i + 1]

            frame_type = determine_frame_type(prev_frame, current_frame, next_frame, adaptive_threshold_sad,
                                              adaptive_threshold_flow, adaptive_threshold_b_frame)
            if (i % 3 == 0) and (frame_type == 'B'):
                frame_types.append('P')
            else:
                frame_types.append(frame_type)

    last_frame_type = determine_last_frame_type(frames[-2], frames[-1], adaptive_threshold_sad, adaptive_threshold_flow)
    frame_types.append(last_frame_type)

    frame_types_str = "".join(frame_types)

    return frame_types, frame_types_str



