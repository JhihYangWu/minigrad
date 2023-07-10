import unittest
from minigrad.tensor import Tensor
import minigrad.optim as optim
from minigrad.utils import get_mnist
import numpy as np

class TestMNIST(unittest.TestCase):
    def test_mnist(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.SGD(model.params(), lr=0.01)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 200, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_momentum(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.Momentum(model.params(), lr=0.01)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 200, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_rmsprop(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.RMSprop(model.params(), lr=0.01)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 200, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_adam_lr_decay(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.Adam(model.params(), lr=0.1, lr_decay=0.5, steps_per_epoch=x_train.shape[0]/256)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 200, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_cnn(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = ConvModel()
        optimizer = optim.Adam(model.params(), lr=0.001)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 100, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

class FullyConnectedModel:
    def __init__(self):
        self.l1 = Tensor.rand_init(784, 128)
        self.l2 = Tensor.rand_init(128, 64)
        self.l3 = Tensor.rand_init(64, 10)
        self.g1 = Tensor(np.random.uniform(size=128))
        self.b1 = Tensor.rand_init(128)
        self.g2 = Tensor(np.random.uniform(size=64))
        self.b2 = Tensor.rand_init(64)

    def params(self):
        return [self.l1, self.l2, self.l3, self.g1, self.b1, self.g2, self.b2]

    def forward(self, x):
        pred = x.matmul(self.l1)
        pred = pred.batchnorm1d(self.g1, self.b1)
        pred = pred.relu().matmul(self.l2)
        pred = pred.batchnorm1d(self.g2, self.b2)
        pred = pred.relu().matmul(self.l3).softmax(dim=1)
        return pred

class ConvModel:
    def __init__(self):
        self.c1 = Tensor.rand_init(8, 1, 3, 3)
        self.c2 = Tensor.rand_init(16, 8, 3, 3)
        self.l1 = Tensor.rand_init(784, 10)

    def params(self):
        return [self.c1, self.c2, self.l1]

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(shape=(-1, 1, 28, 28))
        x = x.pad(pad=(1, 1, 1, 1)).conv2d(self.c1).relu().maxpool2d(kernel_size=(2, 2), stride=2)
        x = x.pad(pad=(1, 1, 1, 1)).conv2d(self.c2).relu().maxpool2d(kernel_size=(2, 2), stride=2)
        x = x.reshape(shape=(batch_size, -1))
        x = x.matmul(self.l1).softmax(dim=1)
        return x

if __name__ == "__main__":
    unittest.main()

