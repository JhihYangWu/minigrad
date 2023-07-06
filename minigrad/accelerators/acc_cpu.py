from minigrad.tensor import Function, register, Tensor
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
        return np.array([x.sum()], dtype=get_float_32_64())

    @staticmethod
    def backward(context, your_grad):
        x = context.safe[0]
        return (your_grad * np.ones(x.shape, dtype=get_float_32_64()),)
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

class Log2(Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return np.log2(x)

    @staticmethod
    def backward(context, your_grad):
        x = context.safe[0]
        return (your_grad / (x * np.log(2)),)
register("log2", Log2)

class Softmax(Function):
    @staticmethod
    def forward(context, x, dim):
        s_x = np.exp(x)
        s_x = s_x / np.sum(s_x, axis=dim).reshape(-1, 1)
        context.save_for_backward(s_x)
        return s_x

    @staticmethod
    def backward(context, your_grad):
        def find_grad_1_dim(s_x, g_s_x):
            l = s_x.shape[0]
            matrix = ((-s_x.dot(s_x.T)) *
                      (1 - np.eye(l)) +
                      (np.eye(l) * (s_x * (1 - s_x))))
            return g_s_x.dot(matrix).astype(get_float_32_64())
        batch_size, num_features = context.safe[0].shape
        parent_grad = np.zeros((batch_size, num_features), dtype=get_float_32_64())
        for i in range(batch_size):
            s_x = context.safe[0][i, :].reshape(1, -1).T
            g_x = find_grad_1_dim(s_x, your_grad[i, :])
            parent_grad[i, :] = g_x
        return (parent_grad,)
register("softmax", Softmax)

def get_float_32_64():
    return np.float64 if Tensor.NEED_PRECISION else np.float32 

