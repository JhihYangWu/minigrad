from minigrad.tensor import Function, register
import numpy as np

class Mul(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(context, your_grad):
        x, y = context.safe
        return (y * your_grad, x * your_grad)
register("mul", Mul)

class Add(Function):
    @staticmethod
    def forward(context, x, y):
        return x + y

    @staticmethod
    def backward(context, your_grad):
        return (your_grad, your_grad)
register("add", Add)

class MatMul(Function):
    @staticmethod
    def forward(context, input, weight):
        context.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(context, your_grad):
        input, weight = context.safe
        input_grad = your_grad.dot(weight.T)
        weight_grad = your_grad.T.dot(input).T
        return (input_grad, weight_grad)
register("matmul", MatMul)

class Sum(Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return np.array([x.sum()], dtype=np.float32)

    @staticmethod
    def backward(context, your_grad):
        x = context.safe[0]
        return (your_grad * np.ones(x.shape, dtype=np.float32),)
register("sum", Sum)

class ReLU(Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(context, your_grad):
        x = context.safe[0]
        return (your_grad * (x >= 0),)
register("relu", ReLU)

