import numpy as np
import matplotlib.pyplot as plt

from network import Network
from tqdm import tqdm


def prepare_train_data(a, b, c, d, num_samples):
    """prepare train data
    Args:
        a (int)
        b (int)
        c (int)
        d (int)
        num_samples (int): the number of points by uniformly sampling from -pi to pi
    Returns:
        (x, y)
    """
    x = np.linspace(-np.pi, np.pi, num_samples)
    y = a * np.cos(b * x) + c * np.sin(d * x)  # y = acos(bx) + csin(dx)

    return x, y


def train(X, y, lr=0.001, epochs=1000):
    """train network on training set
    Args:
        X (np.ndarray): shape is (n,)
        y (np.ndarray): shape is (n,)
        lr (float): learning rate
        epochs (int): max epochs
    Returns:
        Network: trained network
    """
    train_set_size = len(X)
    net = Network(num_layers=4, num_neurons_per_layer=[1, 20, 200, 20, 1])

    for epoch in range(epochs):
        for i in tqdm(range(train_set_size), desc='Epoch {}/{}'.format(epoch + 1, epochs)):
            # forward
            net.forward(X[i].reshape(-1, 1))
            # backward
            net.backward(y[i].reshape(-1, 1), lr)

    return net


def fit(net, X):
    """fit the given function
    Args:
        net (Network): trained network
        X (np.ndarray): shape is (n,)
    Returns:
        y (list): predicted y
    """
    y = []
    for i in range(len(X)):
        # forward
        pred = net.forward(X[i].reshape(-1, 1))
        y.append(pred.item())

    return y


if __name__ == '__main__':
    X, y = prepare_train_data(3, 2, 4, 3, num_samples=300)  # uniformly sampling 300 points from -pi to pi
    trained_net = train(X, y, epochs=3000)
    predicted_y = fit(trained_net, X)

    plt.figure()
    p_true = plt.plot(X, y)
    p_pred = plt.plot(X, predicted_y)
    plt.legend([p_true, p_pred], labels=['$y=3cos2x+4sin3x$', 'forecast curve'], loc='best')
    plt.show()

    print('Average error between predicted curve and true cruve is {}.'.format(np.mean(np.abs(np.array(predicted_y) - y))))
