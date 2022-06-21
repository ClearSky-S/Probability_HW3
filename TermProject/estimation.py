# 2021038131 장준혁

import numpy as np
import matplotlib.pyplot as plt

# a, b, seed 값을 바꿔주세오
a = 1
b = 10
seed = 10
size = 200


def meanSquare(x,y):
    return np.sum((y-a*x-b)**2)/size



if __name__ == "__main__":
    print("a = ", a)
    print("b = ", b)
    np.random.seed(seed)
    x = np.random.normal(4, 5, size)
    y1 = a * x + b + np.random.normal(0, 1, size) * np.exp((-(x - 4) ** 2) / 50)
    y2 = a * x + b + np.random.normal(0, 3, size) * np.exp((-(x - 4) ** 2) / 50)
    print('mean squared difference')
    print('y1 '+str(meanSquare(x,y1)))
    print('y2 '+str(meanSquare(x,y2)))
