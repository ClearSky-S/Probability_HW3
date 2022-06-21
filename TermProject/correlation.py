# 2021038131 장준혁

import numpy as np
import matplotlib.pyplot as plt

# a, b, seed 값을 바꿔주세오
a = 5
b = 10
seed = 10
size = 1000


def cov(x, y):
    return np.sum(x*y) / size - (np.sum(x) / size) * (np.sum(y) / size)


def corr(x: list, y: list):
    return cov(x, y)/(np.std(x)*np.std(y))


if __name__ == "__main__":
    print("a = ", a)
    print("b = ", b)
    np.random.seed(seed)
    x = np.random.normal(4, 5, size)
    y1 = a * x + b + np.random.normal(0, 1, size) * np.exp((-(x - 4) ** 2) / 50)
    y2 = a * x + b + np.random.normal(0, 3, size) * np.exp((-(x - 4) ** 2) / 50)
    print("corr(X, Y1): ", corr(x,y1))
    print("corr(X, Y2): ", corr(x,y2))

    plt.title('a='+str(a)+' b='+str(b)+' σ='+str(1))
    plt.xlabel('X')
    plt.ylabel('Y1')
    plt.scatter(x, y1)
    plt.show()

    plt.title('a='+str(a)+' b='+str(b)+' σ='+str(3))
    plt.xlabel('X')
    plt.ylabel('Y2')
    plt.scatter(x, y2)
    plt.show()
