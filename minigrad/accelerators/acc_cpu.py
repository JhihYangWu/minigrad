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

class Div(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(context, your_grad):
       x, y = context.safe
       return (1/y * your_grad, x * your_grad)
register("div", Div)

class Add(Function):
    @staticmethod
    def forward(context, x, y):
        return x + y

    @staticmethod
    def backward(context, your_grad):
        return (your_grad, your_grad)
register("add", Add)

class Sub(Function):
    @staticmethod
    def forward(context, x, y):
        return x - y

    @staticmethod
    def backward(context, your_grad):
        return (your_grad, -your_grad)
register("sub", Sub)

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

class LeakyReLU(Function):
    @staticmethod
    def forward(context, x, a=0.01):
        context.save_for_backward(x)
        context.save_for_backward(a)
        return np.maximum(x, 0) + a * np.minimum(x, 0)

    @staticmethod
    def backward(context, your_grad):
        x, a = context.safe
        return (your_grad * (1 * (x >= 0) + a * (x < 0)),)
register("leaky_relu", LeakyReLU)

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

class Exp2(Function):
    @staticmethod
    def forward(context, x):
        e_x = np.exp2(x)
        context.save_for_backward(e_x)
        return e_x

    @staticmethod
    def backward(context, your_grad):
        e_x = context.safe[0]
        return (your_grad * e_x * np.log(2),)
register("exp2", Exp2)

class Sin(Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        return np.sin(x)

    @staticmethod
    def backward(context, your_grad):
        x = context.safe[0]
        return (your_grad * np.cos(x),)
register("sin", Sin)

class Sqrt(Function):
    @staticmethod
    def forward(context, x):
        sqrt_x = np.sqrt(x)
        context.save_for_backward(sqrt_x)
        return sqrt_x

    @staticmethod
    def backward(context, your_grad):
        sqrt_x = context.safe[0]
        return (your_grad / (2 * sqrt_x),)
register("sqrt", Sqrt)

class Max(Function):
    @staticmethod
    def forward(context, x):
        context.save_for_backward(x)
        context.save_for_backward(x.argmax())
        return np.array([x.max()], dtype=get_float_32_64())

    @staticmethod
    def backward(context, your_grad):
        x, x_argmax = context.safe
        parent_grad = np.zeros(np.prod(x.shape), dtype=get_float_32_64())
        parent_grad[x_argmax] = your_grad
        parent_grad = parent_grad.reshape(x.shape)
        return (parent_grad,)
register("max", Max)

class Reshape(Function):
    @staticmethod
    def forward(context, x, shape=None):
        context.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(context, your_grad):
        x_shape = context.safe[0]
        return (your_grad.reshape(x_shape),)
register("reshape", Reshape)

class Conv2D(Function):
    @staticmethod
    def forward(context, x, w, stride=1):
        context.save_for_backward(x)
        context.save_for_backward(w)
        context.save_for_backward(stride)
        batch_size, x_channels, x_h, x_w = x.shape
        num_filters, w_channels, w_h, w_w = w.shape
        assert x_channels == w_channels
        out_shape = (batch_size,
                     num_filters,
                     (x_h - w_h) // stride + 1,
                     (x_w - w_w) // stride + 1)
        out = np.zeros(out_shape, dtype=get_float_32_64())
        for i in range(out_shape[2]):
            for j in range(out_shape[3]):
                for k in range(num_filters):
                    filter = w[k, :, :, :]
                    chunk = x[:, :, i*stride:i*stride+w_h, j*stride:j*stride+w_w]
                    out[:, k, i, j] = (chunk * filter).reshape(batch_size, -1).sum(axis=1)
        return out

    def backward(context, your_grad):
        x, w, stride = context.safe
        batch_size, x_channels, x_h, x_w = x.shape
        num_filters, w_channels, w_h, w_w = w.shape
        x_grad = np.zeros(x.shape, dtype=get_float_32_64())
        w_grad = np.zeros(w.shape, dtype=get_float_32_64())
        out_shape = your_grad.shape
        for i in range(out_shape[2]):
            for j in range(out_shape[3]):
                for k in range(num_filters):
                    filter = w[k, :, :, :]
                    chunk = x[:, :, i*stride:i*stride+w_h, j*stride:j*stride+w_w]
                    g_out = your_grad[:, k, i, j].reshape((-1, 1, 1, 1))
                    w_grad[k, :, :, :] += (g_out * chunk).sum(axis=0)
                    x_grad[:, :, i*stride:i*stride+w_h, j*stride:j*stride+w_w] += g_out * filter
        return x_grad, w_grad
register("conv2d", Conv2D)

class Sigmoid(Function):
    @staticmethod
    def forward(context, x):
        s_x = 1 / (1 + np.exp(-x))
        context.save_for_backward(s_x)
        return s_x

    def backward(context, your_grad):
        s_x = context.safe[0]
        return (your_grad * s_x * (1 - s_x),)
register("sigmoid", Sigmoid)

class Tanh(Function):
    @staticmethod
    def forward(context, x):
        t_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        context.save_for_backward(t_x)
        return t_x

    def backward(context, your_grad):
        t_x = context.safe[0]
        return (your_grad * (1 - np.square(t_x)),)
register("tanh", Tanh)

def get_float_32_64():
    return np.float64 if Tensor.NEED_PRECISION else np.float32 

