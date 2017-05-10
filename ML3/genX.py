# -*- coding: utf-8 -*-
#from Guy007 download 03/21 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def draw(x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = x_train[:, 0]
    y = x_train[:, 1]
    z = x_train[:, 2]

    ax.scatter(x, y, z, c=y_train, marker='.')

    plt.title(r'Week 9 Project - Clustering')
    ax.set_xlabel(r'$d_0$')
    ax.set_ylabel(r'$d_1$')
    ax.set_zlabel(r'$d_2$')

    plt.show()


def main(k=5, d=3, n=1000):
    # k: number of clusters
    # d: number of independent variables
    # n: number of training examples to output

    # distribution of clusters (pi)
    pi = np.random.choice([1, 2, 3, 4, 5], size=k)
    print('pi\n', pi)
    pi = pi / sum(pi)

    np.random.seed(0)
    # generating the probability distribution of the clusters
    mu = np.random.randint(low=-20, high=21, size=k * d).reshape(k, d)
    print('mu\n', mu)

    sigma = np.random.randint(low=2, high=15, size=k * d).reshape(k, d)
    print('sigma\n', sigma)

    # generating clusters associated to the n training examples
    y_train = np.random.choice(range(k), size=n, p=pi)

    # generating the n training examples
    x_train = np.zeros((n, d))
    for y in range(k):
        index, = np.where(y_train == y)
        x_train[index, :] = np.random.multivariate_normal(mu[y, :], np.diag(sigma[y, :]), len(index))

    print('x_train\n',x_train[:5, :])

    # saving the training examples to a .csv file
    np.savetxt("X.csv", x_train, delimiter=",")
    np.savetxt("y.csv", y_train, delimiter=",")

    # plotting the n training examples
    draw(x_train, y_train)

if __name__ == "__main__":
    main(k=5, d=3, n=1000)
