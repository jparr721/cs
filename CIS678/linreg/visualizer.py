import matplotlib.pyplot as plt

def visualize():
    x = []
    y = []
    with open('bin/downloads.txt', 'r') as dl:
        for line in dl:
            ll = dl.split(',')
            left, right = ll[0], ll[1])
            x.append(left)
            y.append(right)

    with open('bin/linear.txt', 'r') as lin:
        results = lin.readline().split(' ')
        a, b = results[0], results[1]
        plt.scatter(x, y)
        yfit = [a + b * xi for xi in x]
        plt.plot(x, yfit)

    with open('bin/polynoamia.txt', 'r') as poly:
        results = poly.readline().split(' ')
        print(results)

visualize()
