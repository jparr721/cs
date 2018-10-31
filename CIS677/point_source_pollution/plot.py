import numpy as np
import matplotlib.pyplot as plt
import sys


def visualize():
    path = sys.argv[1]
    data = []

    with open(path) as f:
        content = f.readline()

        data.append([float(x) for x in content.split(' ')])

    plot = np.array(data)
    fig, ax = plt.subplots(nrows=1, sharex=True)
    ax.imshow(plot, cmap='plasma', aspect='auto')

    ax.set_title('Point source pollution')
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


visualize()
