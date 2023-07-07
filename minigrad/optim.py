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
    def __init__(self, model_params, lr=1e-5):
        self.model_params = model_params
        self.lr = lr

    def step(self):
        for weight_t in self.model_params:
            weight_t.data -= self.lr * weight_t.grad.data

class Momentum:
    def __init__(self, model_params, lr=1e-5, beta=0.9):
        # beta=0.9 is moving average of approximately past 10 mini-batches.
        # Momentum allows you to use higher lr than SGD because exploding
        # pos-neg-pos-neg gradients get averaged out.
        self.model_params = model_params
        self.lr = lr
        self.beta = beta

        self.v = [np.zeros_like(weight_t.data) for weight_t in self.model_params]

    def step(self):
        for i in range(len(self.model_params)):
            weight_t = self.model_params[i]
            new_grad = weight_t.grad.data
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * new_grad
            weight_t.data -= self.lr * self.v[i]

class RMSprop:
    def __init__(self, model_params, lr=1e-5, beta2=0.999, eps=1e-8):
        # RMSprop also allows you to use higher lr than SGD because larger
        # gradients get larger denominator and smaller gradients get smaller
        # denominator so same speed every direction.
        self.model_params = model_params
        self.lr = lr
        self.beta2 = beta2
        self.eps = eps

        self.v2 = [np.zeros_like(weight_t.data) for weight_t in self.model_params]

    def step(self):
        for i in range(len(self.model_params)):
            weight_t = self.model_params[i]
            new_grad = weight_t.grad.data
            self.v2[i] = self.beta2 * self.v2[i] + (1 - self.beta2) * (new_grad * new_grad)
            weight_t.data -= self.lr * new_grad / (np.sqrt(self.v2[i]) + self.eps)

class Adam:
    def __init__(self, model_params, lr=1e-5, beta1=0.9, beta2=0.999, eps=1e-8):
        # Adaptive Moment Estimation, basically just Momentum + RMSprop.
        # Can use even higher lr than Momentum or RMSprop.
        self.model_params = model_params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.v1 = [np.zeros_like(weight_t.data) for weight_t in self.model_params]
        self.v2 = [np.zeros_like(weight_t.data) for weight_t in self.model_params]

    def step(self):
        self.t += 1
        for i in range(len(self.model_params)):
            weight_t = self.model_params[i]
            new_grad = weight_t.grad.data
            self.v1[i] = self.beta1 * self.v1[i] + (1 - self.beta1) * new_grad
            self.v2[i] = self.beta2 * self.v2[i] + (1 - self.beta2) * (new_grad * new_grad)
            # Bias correction in exponentially weighted averages. Fix slow start problem.
            v1_corr = self.v1[i] / (1 - self.beta1 ** self.t)
            v2_corr = self.v2[i] / (1 - self.beta2 ** self.t)
            weight_t.data -= self.lr * v1_corr / (np.sqrt(v2_corr) + self.eps)

