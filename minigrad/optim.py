from tqdm import trange
import numpy as np
from minigrad.tensor import Tensor

def train(x_train, y_train, model, optimizer, loss_func, steps, batch_size):
    for i in (t := trange(steps)):
        # Get mini batch to train on.
        sample_indices = np.random.randint(0, x_train.shape[0], size=batch_size)
        x = Tensor(x_train[sample_indices])
        y = Tensor(y_train[sample_indices])
        n_one = Tensor([-1])

        # Forward prop.
        pred = model.forward(x)

        # Calculate loss.
        if loss_func == "CategoricalCrossentropy":
            loss = pred.log2().mul(y).sum().mul(n_one) 

        # Backward prop.
        loss.backward()
        optimizer.step()

        # Calculate accuracy.
        accuracy = -1
        if loss_func == "CategoricalCrossentropy":
            accuracy = (pred.data * y.data).sum() / batch_size

        # Print stats.
        t.set_description("Loss: %.2f | Accuracy: %.2f" % (loss.data, accuracy))

def evaluate(x_test, y_test, model, loss_func, batch_size):
    losses, accuracies = [], []
    i = 0
    m = x_test.shape[0]
    while i + batch_size < m:
        # Get mini batch to test on.
        x = Tensor(x_test[i:i+batch_size])
        y = Tensor(y_test[i:i+batch_size])
        n_one = Tensor([-1])

        # Forward prop.
        pred = model.forward(x)

        # Calculate loss.
        if loss_func == "CategoricalCrossentropy":
            loss = pred.log2().mul(y).sum().mul(n_one) 

        # Calculate accuracy.
        accuracy = -1
        if loss_func == "CategoricalCrossentropy":
            accuracy = (pred.data * y.data).sum() / batch_size

        # Record stats.
        losses.append(loss.data)
        accuracies.append(accuracy)

        i += batch_size

    # Calculate overall loss and accuracy.
    overall_loss = np.array(losses).mean()
    overall_acc = np.array(accuracies).mean()

    return overall_loss, overall_acc

class SGD:
    def __init__(self, model_params, lr=0.00001):
        self.model_params = model_params
        self.lr = lr

    def step(self):
        for weight_t in self.model_params:
            weight_t.data -= self.lr * weight_t.grad.data

