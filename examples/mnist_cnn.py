# Showcase how minigrad can be used to train a CNN to recognize digits in MNIST dataset.

from minigrad.utils import get_mnist
from minigrad.tensor import Tensor
import minigrad.optim as optim
import numpy as np
import cv2

def main():
    x_train, y_train, x_test, y_test = get_mnist()
    model = CNN()
    optimizer = optim.Adam(model.params(), lr=0.001)
    optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", steps=50, batch_size=256)
    show_preds(model, x_test)

class CNN:
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

def show_preds(model, x_test):
    for i in range(x_test.shape[0]):
        x = x_test[i].reshape((1, 784))
        pred = model.forward(Tensor(x)).data
        img_arr = x.reshape((28, 28, 1))
        show_img(img_arr, np.argmax(pred), np.max(pred))

def show_img(img_arr, pred_label, pred_conf):
    img_arr = cv2.resize(img_arr, (800, 800))
    pred_conf *= 100
    cv2.imshow(f"{pred_label} | {pred_conf:.2f}%", img_arr)
    if cv2.waitKey(2000) & 0xFF == ord("q"):
        import sys
        sys.exit()

if __name__ == "__main__":
    main()

