import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def lin_reg():
    frame = pd.read_csv('bin/downloads-test.csv')
    frame = frame.dropna()
    x = frame['hours']
    y = frame['downloads']
    x = list(x)
    y = list(y)

    # train
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    print(f)

    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)

    plt.plot(x, y, 'o', x_new, y_new)
    plt.show()

    # body_reg = linear_model.LinearRegression()
    # body_reg.fit(x, y)
    # plt.scatter(x, y)
    # plt.plot(x, body_reg.predict(x))
    # plt.show()


def visualize():
    x = []
    y = []
    with open('bin/downloads.txt', 'r') as dl:
        for line in dl:
            ll = line.split(',')
            left, right = (ll[0], ll[1])
            x.append(left)
            y.append(right)

    with open('bin/linear.txt', 'r') as lin:
        results = lin.readline().split(' ')
        x = [int(xi) for xi in x]
        y = [float(yi) for yi in y]
        a, b = results[0], results[1]
        a, b = float(a)/100, float(b)/200
        plt.scatter(x, y)
        yfit = [a + b * xi for xi in x]
        plt.plot(x, yfit)
        plt.show()

    with open('bin/polynomial.txt', 'r') as poly:
        results = poly.readline().split(' ')
        print(results)


lin_reg()
