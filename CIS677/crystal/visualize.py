import matplotlib.pyplot as plt
import numpy as np

PATH = './bin/output.txt'


def visualize(path):
    data_buffer = []

    with open(path) as f:
        content = f.readlines()

        for line in content:
            data_buffer.append([int(x) for x in line.split()])

    plottable = np.array(data_buffer)

    plt.imshow(plottable)

    plt.show()


visualize(PATH)
