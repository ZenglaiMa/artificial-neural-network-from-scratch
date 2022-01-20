import numpy as np

from network import Network
from tqdm import tqdm


def process_data(train_data_path, test_data_path):
    """read and process the data, the MNIST dataset is given by csv format
    Args:
        train_data_path (string)
        test_data_path (string)
    Returns:
        train_pixel (np.ndarray): shape is (60000, 784)
        train_label (np.ndarray): shape is (60000,)
        test_pixel (np.ndarray): shape is (10000, 784)
        test_label (np.ndarray): shape is (10000,)
    """
    train_pixel, train_label = [], []
    with open(train_data_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip().split(',')
            train_pixel.append(line[1:])
            train_label.append(line[0])

    test_pixel, test_label = [], []
    with open(test_data_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip().split(',')
            test_pixel.append(line[1:])
            test_label.append(line[0])

    train_pixel = np.array(train_pixel, dtype='int32')
    train_label = np.array(train_label, dtype='int32')
    test_pixel = np.array(test_pixel, dtype='int32')
    test_label = np.array(test_label, dtype='int32')

    return train_pixel, train_label, test_pixel, test_label


def train_and_test(train_pixel, train_label, test_pixel, test_label, lr=0.1, epochs=10):
    """train network on training set and test it on testing set
    Args:
        returned values from process_data() function
        lr (float): learning rate
        epochs (int): max epochs
    """
    correct_count = 0  # use to count the number of correctly classified images on testing set
    train_set_size = len(train_pixel)
    test_set_size = len(test_pixel)
    net = Network(num_layers=3, num_neurons_per_layer=[784, 200, 50, 10])  # last layer hasn't been activated

    print('begin training...')
    for epoch in range(epochs):
        for i in tqdm(range(train_set_size), desc='Epoch {}/{}'.format(epoch + 1, epochs)):
            x = train_pixel[i].reshape(-1, 1) / 255.0  # do normalization: (x - x_min) / (x_max - x_min), x_min = 0 and x_max = 255
            y = np.zeros((10, 1))
            y[train_label[i]] = 1
            # forward
            net.forward(x)
            # backward
            net.backward(y, lr)

    print('\nbegin testing...')
    for i in tqdm(range(test_set_size)):
        x = test_pixel[i].reshape(-1, 1) / 255.0  # do normalization: (x - x_min) / (x_max - x_min), x_min = 0 and x_max = 255
        pred = net.forward(x)
        if pred.argmax() == test_label[i]:
            correct_count += 1

    print('\nAccuracy on test set is {}.\n'.format(correct_count / test_set_size * 100))


if __name__ == '__main__':
    train_pixel, train_label, test_pixel, test_label = process_data('./data/mnist_train.csv', './data/mnist_test.csv')
    train_and_test(train_pixel, train_label, test_pixel, test_label)
