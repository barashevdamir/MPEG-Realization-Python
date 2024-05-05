import numpy as np
import scipy
import scipy.ndimage



def lucas_kanade(Image1: np.ndarray, Image2: np.ndarray, N, tau=1e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет оптический поток двух изображений Image1 и Image2 по методу Лукаса-Канады.

    :param Image1: Первое изображение
    :param Image2: Второе изображение
    :param N: Размер окна
    :param tau: Скорость обучения
    :return: Оптический поток между Image1 и Image2
    """
    Image1 = Image1 / 255
    Image2 = Image2 / 255
    image_shape = Image1.shape
    half_window_size = N // 2

    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    Ix = scipy.ndimage.convolve(input=Image1, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=Image1, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=Image2, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(input=Image1,
                                                                                                         weights=-kernel_t,
                                                                                                         mode="nearest")
    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    for row_ind in range(half_window_size, image_shape[0] - half_window_size):
        for col_ind in range(half_window_size, image_shape[1] - half_window_size):
            Ix_windowed = Ix[
                          row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1,
                          ].flatten()
            Iy_windowed = Iy[
                          row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1,
                          ].flatten()
            It_windowed = It[
                          row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1,
                          ].flatten()

            A = np.asarray([Ix_windowed, Iy_windowed]).reshape(-1, 2)
            b = np.asarray(It_windowed)

            A_transpose_A = np.transpose(A) @ A

            A_transpose_A_eig_vals, _ = np.linalg.eig(A_transpose_A)
            A_transpose_A_min_eig_val = np.min(A_transpose_A_eig_vals)

            if A_transpose_A_min_eig_val < tau:  # To threshold noise
                continue

            A_transpose_A_PINV = np.linalg.pinv(A_transpose_A)
            w = A_transpose_A_PINV @ np.transpose(A) @ b

            u[row_ind, col_ind], v[row_ind, col_ind] = w

    flow = [u, v]
    I = [Ix, Iy, It]

    return flow, I

def compute_gradients(Image1: np.ndarray, Image2: np.ndarray) -> list[np.ndarray]:
    """
    Вычисляет градиенты двух изображений Image1 и Image2.

    :param Image1: Первое изображение
    :param Image2: Второе изображение
    :return: Градиенты двух изображений
    """

    Image1 = Image1 / 255
    Image2 = Image2 / 255

    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    Ix = scipy.ndimage.convolve(input=Image1, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=Image1, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=Image2, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=Image1, weights=-kernel_t, mode="nearest"
    )

    I = [Ix, Iy, It]

    return I


def horn_schunck(Image1: np.ndarray, Image2: np.ndarray, lamda: float) -> list[np.ndarray]:
    """
    Вычисляет оптический поток между двумя изображениями Image1 и Image2.

    :param Image1: Первое изображение
    :param Image2: Второе изображение
    :param lamda: Штраф
    :return: Оптический поток между Image1 и Image2
    """

    u = np.zeros([Image1.shape[0], Image1.shape[1]])
    v = np.zeros([Image1.shape[0], Image1.shape[1]])

    [Ix, Iy, It] = compute_gradients(Image1, Image2)

    kernel = np.array([[0, 1 / 4, 0], [1 / 4, 0, 1 / 4], [0, 1 / 4, 0]], dtype=np.float32) # Optical flow averaging kernel

    for _ in range(500):
        u_avg = scipy.ndimage.convolve(input=u, weights=kernel, mode="nearest")
        v_avg = scipy.ndimage.convolve(input=v, weights=kernel, mode="nearest")

        grad = (Ix * u_avg + Iy * v_avg + It) / (lamda ** 2 + Ix ** 2 + Iy ** 2)

        u = u_avg - lamda * Ix * grad
        v = v_avg - lamda * Iy * grad

    flow = [u, v]
    I = [Ix, Iy, It]

    return flow, I