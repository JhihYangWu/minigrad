import unittest
from minigrad.tensor import Tensor
import minigrad.optim as optim
import numpy as np

class TestMNIST(unittest.TestCase):
    def test_mnist(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.SGD(model.params())
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 1000, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_momentum(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.Momentum(model.params(), lr=0.0001)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 1000, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

    def test_mnist_rmsprop(self):
        x_train, y_train, x_test, y_test = get_mnist()
        model = FullyConnectedModel()
        optimizer = optim.RMSprop(model.params(), lr=0.0001)
        optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", 1000, 256)
        test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", 256)
        print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
        assert test_acc >= 0.9

class FullyConnectedModel:
    def __init__(self):
        self.l1 = Tensor.rand_init(784, 128)
        self.l2 = Tensor.rand_init(128, 64)
        self.l3 = Tensor.rand_init(64, 10)

    def params(self):
        return [self.l1, self.l2, self.l3]

    def forward(self, x):
        return x.matmul(self.l1).relu().matmul(self.l2).relu().matmul(self.l3).softmax(dim=1)

def get_mnist():
    def fetch(url):
        import requests, gzip, os, hashlib, tempfile
        fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                data = f.read() 
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    x_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    x_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    x_train = x_train.reshape((-1, 784)).astype(np.float32)
    x_test = x_test.reshape((-1, 784)).astype(np.float32)
    y = np.zeros((len(y_train), 10), dtype=np.float32)
    y[range(y.shape[0]), y_train] = 1
    y_train = y
    y = np.zeros((len(y_test), 10), dtype=np.float32)
    y[range(y.shape[0]), y_test] = 1
    y_test = y

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    unittest.main()

