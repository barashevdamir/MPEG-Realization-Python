import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

scale = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# Создаем пример данных
zip = np.array([
    0.8745945596910865,
    1.2038247303316496,
    1.6517169410214199,
    2.22448905163136,
    2.940907728667771,
    3.773998268607415,
    4.6476356577758455,
    5.473569684082767,
    6.237018137287063,
    6.855843660358378,
    7.2682481311243885
])

# Создаем функцию интерполяции
interpolator = interp1d(scale, zip, kind='cubic')

# Вычисляем новые значения x для интерполяции
new_x = np.linspace(scale.min(), scale.max(), 128)

# Вычисляем новые значения y с помощью интерполяции
new_y = interpolator(new_x)

# Построение графика
plt.plot(scale, zip, 'o', new_x, new_y, '-')
plt.xlabel('Масштаб качества')
plt.ylabel('Коэффициент сжатия')
plt.title('')
plt.grid(True)
plt.show()

psnr = np.array([
    48.9,
    47.21,
    45.02,
    42.53,
    40.13,
    38.18,
    36.55,
    34.85,
    32.77,
    31.01,
    28.82
])

# Создаем функцию интерполяции
interpolator = interp1d(scale, psnr, kind='cubic')

# Вычисляем новые значения x для интерполяции
new_x = np.linspace(scale.min(), scale.max(), 128)

# Вычисляем новые значения y с помощью интерполяции
new_y = interpolator(new_x)

# Построение графика
plt.plot(scale, psnr, 'o', new_x, new_y, '-')
plt.xlabel('Масштаб качества')
plt.ylabel('PSNR')
plt.title('')
plt.grid(True)
plt.show()

ssim = np.array([
    1,
    0.99,
    0.99,
    0.98,
    0.97,
    0.95,
    0.93,
    0.9,
    0.84,
    0.77,
    0.71
])

# Создаем функцию интерполяции
interpolator = interp1d(scale, ssim, kind='cubic')

# Вычисляем новые значения x для интерполяции
new_x = np.linspace(scale.min(), scale.max(), 128)

# Вычисляем новые значения y с помощью интерполяции
new_y = interpolator(new_x)

# Построение графика
plt.plot(scale, ssim, 'o', new_x, new_y, '-')
plt.xlabel('Масштаб качества')
plt.ylabel('SSIM')
plt.title('')
plt.grid(True)
plt.show()

degradation = np.array([
    -0.68,
    -0.34,
    0.24,
    1.16,
    2.33,
    3.75,
    5.67,
    8.89,
    14.63,
    21.68,
    28.39
])

# Создаем функцию интерполяции
interpolator = interp1d(scale, degradation, kind='cubic')

# Вычисляем новые значения x для интерполяции
new_x = np.linspace(scale.min(), scale.max(), 128)

# Вычисляем новые значения y с помощью интерполяции
new_y = interpolator(new_x)

# Построение графика
plt.plot(scale, degradation, 'o', new_x, new_y, '-')
plt.xlabel('Масштаб качества')
plt.ylabel('Коэффициент деградации (%)')
plt.title('')
plt.grid(True)
plt.show()