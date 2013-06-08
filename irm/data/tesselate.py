import numpy as np

def grid(width_n, height_n, width_sep, height_sep):
    N = width_n * height_n
    data = np.zeros((N, 3), dtype=np.float32)
    pos = 0

    for xi in range(width_n):
        for yi in range(height_n):
            data[pos] = (xi * width_sep, yi * height_sep, 0.0)
            pos += 1
    return data

def create_triangle():
    pass

def create_hex():
    pass


