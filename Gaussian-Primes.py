import math
import matplotlib.pyplot as plt


# We aim to plot Gaussian primes
# We try using an object-oriented approach

import numpy as np


def isprime(number):
    for n in range(2, int(math.sqrt(number)) + 1):
        if number % n == 0:
            return False
    return True


class Complex:
    def __init__(self, real, imag):
        self.real_part = real
        self.imaginary_part = imag

    def norm_squared(self):
        return self.real_part**2 + self.imaginary_part**2


def isgaussprime(x, y):
    z = Complex(x, y)
    rez, imz = z.real_part, z.imaginary_part
    if rez != 0 and imz != 0:
        if isprime(z.norm_squared()):
            return True
        else:
            return False
    elif imz == 0 and rez != 0:
        if isprime(abs(rez)) and (abs(rez)) % 4 == 3:
            return True
    elif rez == 0 and imz != 0:
        if isprime(abs(imz)) and (abs(imz)) % 4 == 3:
            return True
    else:
        return False

real_values, imag_values = [], []
for current_real in range(-100, 101):
    for current_imag in range(-100, 101):
        if Complex(current_real, current_imag).norm_squared() <= 10000:
            if isgaussprime(current_real, current_imag):
                real_values.append(current_real)
                imag_values.append(current_imag)

ax = plt.axes()
ax.set_facecolor("black")
plt.scatter(real_values, imag_values, s = 1, c = 'green')
x = np.linspace(-100,100,100)
y1, y2 = x, -x
plt.plot(x, y1, '-y', linewidth=0.5), plt.plot(x, y2, '-y', linewidth=0.5)
plt.axhline(y=0, color='y', linewidth = 0.5, linestyle='-')
plt.axvline(x=0, color='y', linewidth = 0.5, linestyle='-')
plt.axis('equal')  # axis are scaled equally
plt.xlabel('Real'), plt.ylabel('Imaginary')
plt.show()
