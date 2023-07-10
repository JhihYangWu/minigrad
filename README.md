# minigrad
A toy project for me to learn more about the underlying operations of neural networks and various deep learning projects. Just like [tinygrad](https://github.com/tinygrad/tinygrad), minigrad is a deep learning framework designed to be simple and support both inference and training.

---
## Setup
```sh
git clone https://github.com/JhihYangWu/minigrad.git
cd minigrad
pip3 install -e .
```

---
## Train a CNN
```py
from minigrad.utils import get_mnist
from minigrad.tensor import Tensor
import minigrad.optim as optim

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

x_train, y_train, x_test, y_test = get_mnist()
model = CNN()
optimizer = optim.Adam(model.params(), lr=0.001)
optim.train(x_train, y_train, model, optimizer, "CategoricalCrossentropy", steps=50, batch_size=256)
```

---
## Final Notes
minigrad is not meant to be used in production as it is very slow and may be unreliable. However, it is great for learning how neural networks and all the optimization techniques work from scratch. Lastly, I would like to thank George Hotz and Andrew Ng as I wouldn't have been able to create this project without them.
