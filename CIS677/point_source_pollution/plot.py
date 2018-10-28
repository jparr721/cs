import numpy as np
import matplotlib.pyplot as plt

PATH = './output.txt'


def visualize(path):
    data = []

    with open(path) as f:
        content = f.readline()

        data.append([float(x) for x in content.split(' ')])

    plot = np.array(data)
    fig, ax = plt.subplots()
    _ = ax.imshow(plot)

    ax.set_title('Point source pollution')
    fig.tight_layout()
    plt.show()


visualize(PATH)
