import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
import imageio

def sec2timestr(sec):
    """ Конвертирует секунды в удобочитаемый формат времени. """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f'{h:02}:{m:02}:{s:05.2f}'

def mpegproj():
    print("\nMPEG Project\n")

    nf = 0  # Number of frames to process, 0 = process the entire movie

    print(f'{nf} frames\n')

    # Loading the video
    filename = "path_to_video_file.mp4"  # Specify the path to your video file
    mov = getmov(filename, nf)

    # Encoding
    start_time = time.time()
    mpeg = encmov(mov)
    encode_time = time.time() - start_time
    print(f'Encode time: {sec2timestr(encode_time)}\n')

    # Decoding
    start_time = time.time()
    mov2 = decmpeg(mpeg)
    decode_time = time.time() - start_time
    print(f'Decode time: {sec2timestr(decode_time)}\n')

    # Saving the results
    np.savez('lastmov.npz', mov=mov, mpeg=mpeg, mov2=mov2)
    np.savez('mpeg.npz', mpeg=mpeg)
    data = np.load('lastmov.npz')
    framesmov = data['mov']
    framesmov2 = data['mov2']
    data.close()
    mov_folder = 'mov'
    mov2_folder = 'mov2'
    if not os.path.exists(mov_folder):
        os.makedirs(mov_folder)
    if not os.path.exists(mov_folder):
        os.makedirs(mov2_folder)
    for i in range(framesmov.shape[3]):
        frame = framesmov2[:, :, :, i]
        # Указываем путь и формат файла. Например, сохраняем в формате JPEG.
        filename = os.path.join(mov_folder, f'frame_{i + 1}.jpg')
        imageio.imwrite(filename, frame[:, :, ::-1])
        print(f'Saved {filename}')
    for i in range(framesmov2.shape[3]):
        frame = framesmov2[:, :, :, i]
        # Указываем путь и формат файла. Например, сохраняем в формате JPEG.
        filename = os.path.join(mov2_folder, f'frame_{i + 1}.jpg')
        imageio.imwrite(filename, frame[:, :, ::-1])
        print(f'Saved {filename}')


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
    # for i in range(1, 12):
    for i in range(21, 30):
    # for i in range(65, 96):
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

def getmov(filename, nf):
    """
    Загружает видео из файла.

    Аргументы:
    filename -- имя файла видео.
    nf -- количество кадров для загрузки, если nf == 0, загружаются все кадры.

    Возвращает:
    movdata -- данные видео в формате numpy массива.
    """

    folder_path = 'data/frames/'
    height, width = 1920, 1080
    frames = extract_frames(folder_path, height, width)

    # Преобразуем список кадров в 4D numpy массив
    movdata = np.stack(frames, axis=3)

    return movdata

def encmov(mov):
    """
    Кодирует видеоролик, используя заданный шаблон типов кадров.

    Аргументы:
    mov -- входной видеоролик в формате RGB.

    Возвращает:
    mpeg -- список с закодированными данными каждого кадра.
    """
    # Шаблон типов кадров
    fpat = 'IPPPP'  # можно выбрать другой шаблон, например 'I' или динамически сгенерированный

    mpeg = []
    pf = None

    # Перебор кадров
    num_frames = mov.shape[3]
    for i in range(num_frames):
        # Получаем кадр
        f = mov[:, :, :, i].astype(np.float32)

        # Конвертация кадра в YCbCr
        f = rgb2ycc(f)

        # Получаем тип кадра из шаблона
        ftype = fpat[i % len(fpat)]

        # Кодирование кадра
        encoded_frame, pf = encframe(f, ftype, pf)

        # Сохраняем результат
        mpeg.append(encoded_frame)

        # Отображение прогресса (опционально, может потребовать реализации функции progressbar)
        print(f"Encoding frame {i + 1}/{num_frames}...")

    return mpeg

def encframe(f, ftype, pf):
    """
    Кодирует кадр, разбивая его на макроблоки и кодируя каждый макроблок.

    Аргументы:
    f -- текущий кадр для кодирования.
    ftype -- тип кадра ('I' или 'P').
    pf -- предыдущий кадр.

    Возвращает:
    mpeg -- массив со структурами данных для каждого макроблока.
    df -- декодированный кадр после кодирования всех макроблоков.
    """
    M, N, _ = f.shape
    mbsize = (M // 16, N // 16)
    mpeg = np.empty(mbsize, dtype=object)
    df = np.zeros_like(f)

    # Яркостная компонента предыдущего кадра
    if pf is not None:
        pfy = pf[:, :, 0]
    else:
        pfy = np.zeros_like(f[:, :, 0])


    # Перебор макроблоков
    for m in range(mbsize[0]):
        for n in range(mbsize[1]):
            # Вычисляем координаты текущего макроблока
            x = 16 * m
            y = 16 * n
            x_range = slice(x, x + 16)
            y_range = slice(y, y + 16)

            # Кодируем макроблок
            mpeg[m, n], df[x_range, y_range, :] = encmacroblock(
                f[x_range, y_range, :], ftype, pf, pfy, x, y)

    return mpeg, df

def getblocks(mb):
    """
    Извлекает блоки из макроблока.

    Аргументы:
    mb -- входной макроблок.

    Возвращает:
    b -- массив блоков 8x8 для дальнейшей обработки.
    """
    b = np.zeros((8, 8, 6))

    # Четыре блока яркости
    b[:, :, 0] = mb[0:8, 0:8, 0]
    b[:, :, 1] = mb[0:8, 8:16, 0]
    b[:, :, 2] = mb[8:16, 0:8, 0]
    b[:, :, 3] = mb[8:16, 8:16, 0]

    # Два подвыборочных блока цветности
    b[:, :, 4] = 0.25 * (mb[0:16:2, 0:16:2, 1] + mb[0:16:2, 1:16:2, 1] +
                         mb[1:16:2, 0:16:2, 1] + mb[1:16:2, 1:16:2, 1])
    b[:, :, 5] = 0.25 * (mb[0:16:2, 0:16:2, 2] + mb[0:16:2, 1:16:2, 2] +
                         mb[1:16:2, 0:16:2, 2] + mb[1:16:2, 1:16:2, 2])

    return b

def rgb2ycc(rgb):
    """
    Конвертация из RGB в YCbCr.

    Аргументы:
    rgb -- входное изображение в RGB формате.

    Возвращает:
    ycc -- изображение в YCbCr формате.
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
    ycc = np.dot(m, rgb.T)
    ycc = ycc + np.array([[0], [0.5], [0.5]])  # Добавляем смещение к Cb и Cr

    # Возвращаем к исходному размеру
    ycc = ycc.T.reshape(nr, nc, c)
    return ycc


def encmacroblock(mb, ftype, pf, pfy, x, y):
    """
    Кодирование макроблока.

    Аргументы:
    mb -- текущий макроблок.
    ftype -- тип кадра ('I' или 'P').
    pf -- предыдущий кадр.
    pfy -- яркостная компонента предыдущего кадра.
    x, y -- координаты начала макроблока.

    Возвращает:
    mpeg -- структура данных с информацией о макроблоке.
    dmb -- декодированный макроблок.
    """
    # Квантовые матрицы
    q1, q2 = qintra(), qinter()[1]  # используем вторую матрицу из qinter

    # Масштабирование качества
    scale = 31

    # Инициализация структуры MPEG
    mpeg = {
        'type': 'I',
        'mvx': 0,
        'mvy': 0,
        'scale': [scale] * 6,
        'coef': np.zeros((8, 8, 6))  # предположим размер блока 8x8 и 6 блоков
    }

    # Нахождение векторов движения для P-кадров
    if ftype == 'P':
        mpeg['type'] = 'P'
        mpeg, emb = getmotionvec(mpeg, mb, pf, pfy, x, y)
        mb = emb  # используем блок ошибки для кодирования
        q = q2
    else:
        q = q1

    # Получение блоков яркости и цветности
    b = getblocks(mb)

    # Кодирование блоков
    for i in range(6):
        coef = dct2(b[:, :, i])
        mpeg['coef'][:, :, i] = np.round(8 * coef / (scale * q))

    # Декодирование этого макроблока для использования в будущем P-кадре
    dmb = decmacroblock(mpeg, pf, x, y)

    return mpeg, dmb

def getmotionvec(mpeg, mb, pf, pfy, x, y):
    """
    Получение векторов движения для макроблока.

    Аргументы:
    mpeg -- словарь для хранения векторов движения.
    mb -- текущий макроблок.
    pf -- предыдущий кадр.
    pfy -- яркостная компонента предыдущего кадра.
    x, y -- координаты начала текущего макроблока.

    Возвращает:
    mpeg -- обновленный словарь с векторами движения.
    emb -- блок ошибок.
    """
    # Работаем только с яркостной компонентой
    mby = mb[:, :, 0]
    M, N = pfy.shape

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

            sad = np.sum(np.abs(mby - pfy[tx, ty]))

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
    emb = mb - pf[slice(x + mvx, x + mvx + 16), slice(y + mvy, y + mvy + 16), :]

    return mpeg, emb

def decmpeg(mpeg):
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
        f = decframe(mpeg[i], pf)

        # Сохранение предыдущего фрейма
        pf = f.copy()

        # Конвертация фрейма из YCbCr в RGB
        f = ycc2rgb(f)

        # Проверка, что значения пикселей находятся в диапазоне [0, 255]
        f = np.clip(f, 0, 255)

        # Сохранение фрейма в массив видеоролика
        mov[:,:,:,i] = f.astype(np.uint8)

        # (Опционально) Отображение прогресса
        # Это можно сделать, например, используя библиотеку tqdm или просто печатая статус
        print(f"Processing frame {i+1}/{len(mpeg)}...")

    return mov

def decframe(mpeg, pf):
    """
    Декодирует кадр из MPEG потока.

    Аргументы:
    mpeg -- массив объектов или словарей, каждый из которых содержит информацию для одного макроблока.
    pf -- предыдущий кадр, используемый для предсказания в P-кадрах.

    Возвращает:
    fr -- декодированный кадр в виде массива.
    """
    # Размеры массива макроблоков
    mbsize = mpeg.shape
    M = 16 * mbsize[0]
    N = 16 * mbsize[1]

    # Инициализируем кадр нулями
    fr = np.zeros((M, N, 3))

    # Перебор макроблоков
    for m in range(mbsize[0]):
        for n in range(mbsize[1]):
            # Вычисляем координаты в кадре для текущего макроблока
            x = slice(16 * m, 16 * (m + 1))
            y = slice(16 * n, 16 * (n + 1))

            # Декодирование макроблока и помещение его в кадр
            fr[x, y, :] = decmacroblock(mpeg[m, n], pf, 16 * m, 16 * n)

    return fr

def putblocks(b):
    """
    Собирает блоки DCT в один макроблок.

    Аргументы:
    b -- массив блоков DCT, где b[:,:,i] представляет i-й блок.

    Возвращает:
    mb -- макроблок собранный из входных блоков.
    """
    # Создаем пустой макроблок
    mb = np.zeros((16, 16, 3))

    # Четыре блока яркости
    mb[0:8, 0:8, 0] = b[:,:,0]   # Верхний левый
    mb[0:8, 8:16, 0] = b[:,:,1]  # Верхний правый
    mb[8:16, 0:8, 0] = b[:,:,2]  # Нижний левый
    mb[8:16, 8:16, 0] = b[:,:,3] # Нижний правый

    # Два подвыборочных блока цветности
    z = np.array([[1, 1], [1, 1]])
    mb[:,:,1] = np.kron(b[:,:,4], z)  # Кронекер для Cb
    mb[:,:,2] = np.kron(b[:,:,5], z)  # Кронекер для Cr

    return mb

def ycc2rgb(ycc):
    """
    Конвертация из YCbCr в RGB.

    Аргументы:
    ycc -- входное изображение в YCbCr формате.

    Возвращает:
    rgb -- изображение в RGB формате.
    """
    # Матрица преобразования из YCbCr в RGB
    m = np.array([[ 0.299,     0.587,     0.144],
                  [-0.168736, -0.331264,  0.5],
                  [ 0.5,      -0.418688, -0.081312]])
    m = np.linalg.inv(m)  # Вычисляем обратную матрицу

    # Получаем размеры изображения
    nr, nc, c = ycc.shape

    # Переформатирование массива для матричного умножения
    ycc = ycc.reshape(nr*nc, 3)

    # Преобразование цветового кодирования с коррекцией смещения
    rgb = ycc - np.array([0, 0.5, 0.5])
    rgb = np.dot(rgb, m.T)  # Применяем матрицу преобразования

    # Возвращаем к исходному размеру
    rgb = rgb.reshape(nr, nc, c)
    return rgb


def decmacroblock(mpeg, pf, x, y):
    """
    Декодирование макроблока из MPEG потока.

    Аргументы:
    mpeg -- словарь или объект с данными MPEG (тип кадра, векторы движения, коэффициенты, масштабы).
    pf -- предыдущий кадр (для использования в предсказании).
    x, y -- координаты начала макроблока в предыдущем кадре.

    Возвращает:
    mb -- декодированный макроблок.
    """
    # Инициализация матриц квантования
    q1, q2 = qintra(), qinter()[1]  # Предположим, что qinter возвращает матрицу

    mb = np.zeros((16, 16, 3))

    # Предсказание с использованием векторов движения
    if mpeg['type'] == 'P':
        mb = pf[x + mpeg['mvx'] : x + mpeg['mvx'] + 16, y + mpeg['mvy'] : y + mpeg['mvy'] + 16, :]
        q = q2
    else:
        q = q1

    # Декодирование блоков
    b = np.zeros((8, 8, 6))  # Предполагаем, что есть 6 блоков (например, YCbCr)
    for i in range(6):
        coef = mpeg['coef'][:, :, i] * (mpeg['scale'][i] * q) / 8
        b[:, :, i] = idct2(coef)

    # Конструкция макроблока
    mb += putblocks(b)  # предполагается, что putblocks правильно собирает блоки

    return mb

def qinter():
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

def qintra():
    """
    Таблица квантования для I-кадров.

    Возвращает:
    q -- таблица квантования в виде numpy массива.
    """
    q = np.array([
        [ 8, 16, 19, 22, 26, 27, 29, 34],
        [16, 16, 22, 24, 27, 29, 34, 37],
        [19, 22, 26, 27, 29, 34, 34, 38],
        [22, 22, 26, 27, 29, 34, 37, 40],
        [22, 26, 27, 29, 32, 35, 40, 48],
        [26, 27, 29, 32, 35, 40, 48, 58],
        [26, 27, 29, 34, 38, 46, 56, 69],
        [27, 29, 35, 38, 46, 56, 69, 83]
    ])
    return q

def dct2(x):
    """
    Выполнение двумерного дискретного косинусного преобразования (DCT).

    Аргументы:
    x -- входная матрица (numpy array).

    Возвращает:
    y -- матрица после применения DCT.
    """
    # Применяем DCT сначала к столбцам, затем к строкам
    y = dct(dct(x.T, norm='ortho').T, norm='ortho')
    return y

def dctmtx(n):
    """
    Генерация матрицы DCT размера nxn.

    Аргументы:
    n -- размер матрицы.

    Возвращает:
    матрица DCT размера nxn.
    """
    return dct(np.eye(n), norm='ortho')

def dct2_matrix_method(x):
    """
    Выполнение 2D DCT с использованием матрицы DCT для ускорения вычислений.

    Аргументы:
    x -- входная матрица (numpy array).

    Возвращает:
    y -- матрица после применения DCT.
    """
    # Предполагаем, что размер блока равен 8x8
    d = dctmtx(8)
    y = np.dot(d, np.dot(x, d.T))
    return y

def idct2(x):
    """
    Выполнение двумерного обратного дискретного косинусного преобразования (IDCT).

    Аргументы:
    x -- входная матрица (numpy array).

    Возвращает:
    y -- матрица после применения IDCT.
    """
    # Вычисление IDCT по столбцам, затем по строкам
    y = idct(idct(x.T, norm='ortho').T, norm='ortho')
    return y

mpegproj()