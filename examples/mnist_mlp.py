# Showcase how minigrad can be used to train a MLP to recognize digits in MNIST dataset.

from minigrad.utils import get_mnist
from minigrad.tensor import Tensor
import minigrad.optim as optim
import numpy as np
import cv2

def main():
    x_train, y_train, x_test, y_test = get_mnist()
    model = MLP()
    optimizer = optim.Adam(model.params(), lr=0.01)
    optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", steps=200, batch_size=512)
    test_loss, test_acc = optim.evaluate(x_test, y_test, model, "CategoricalCrossentropy", batch_size=512)
    print("Test Loss: %.2f | Test Accuracy: %.2f" % (test_loss, test_acc))
    show_preds(model, x_test)

class MLP:
    def __init__(self):
        self.l1 = Tensor.rand_init(784, 128)
        self.l2 = Tensor.rand_init(128, 64)
        self.l3 = Tensor.rand_init(64, 10)
        self.g1 = Tensor(np.random.uniform(size=128).astype(np.float32))
        self.b1 = Tensor.rand_init(128)
        self.g2 = Tensor(np.random.uniform(size=64).astype(np.float32))
        self.b2 = Tensor.rand_init(64)

    def params(self):
        return [self.l1, self.l2, self.l3, self.g1, self.b1, self.g2, self.b2]

    def forward(self, x):
        return x.matmul(self.l1).batchnorm1d(self.g1, self.b1).relu().matmul(self.l2).batchnorm1d(self.g2, self.b2).relu().matmul(self.l3).softmax(dim=1)

def show_preds(model, x_test):
    preds = model.forward(Tensor(x_test)).data
    for i in range(x_test.shape[0]):
        pred = preds[i]
        img_arr = x_test[i].reshape((28, 28, 1))
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

