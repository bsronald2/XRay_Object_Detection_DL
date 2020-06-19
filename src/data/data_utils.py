import matplotlib.pyplot as plt


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()


def create_chunk(l, n):
    return [l[i: i + n] for i in range(0, len(l), n)]
