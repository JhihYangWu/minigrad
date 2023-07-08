from functools import partialmethod
import numpy as np

class Tensor:
    MUTE_FLOAT_WARNING = False
    NEED_PRECISION = False

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError("Can only construct Tensor with lists or numpy arrays.")

        if data.dtype != np.float32 and not Tensor.MUTE_FLOAT_WARNING:
            print(f"Warning: Tensor with shape {data.shape} isn't float32.")
            Tensor.MUTE_FLOAT_WARNING = True

        self.data = data
        self.grad = None
        self._context = None  # Used for autograd.

    def __repr__(self):
        return f"Tensor with data {self.data} and grad {self.grad}"

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def rand_init(*shape):
        vals = np.random.uniform(-1.0, 1.0, size=shape) / np.sqrt(np.prod(shape))
        return Tensor(vals.astype(np.float32))

    def backward(self, user_called=True):
        def zero_grads(t):
            if t.grad is None:
                t.grad = Tensor(np.zeros(t.data.shape, dtype=t.data.dtype))
            else:
                t.grad.data[:] = 0
            if t._context is not None:
                for p_t in t._context.parents:
                    zero_grads(p_t)
        def find_topological_order(t, ordered_list, in_list):
            if t._context is not None:
                for p_t in t._context.parents:
                    find_topological_order(p_t, ordered_list, in_list)
            if t not in in_list:
                ordered_list.append(t)
                in_list.add(t)
    
        if self._context is None:
            return  # You are root node, no need to calculate more grads.

        if user_called:
            assert self.data.shape == (1,)  # Make sure loss is a single number.
            zero_grads(self)
            self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype))
            ordered_list = []
            find_topological_order(self, ordered_list, set())
            for t in reversed(ordered_list):
                t.backward(user_called=False)

        parent_grads = self._context.func_applied.backward(self._context, self.grad.data)
        for p, g in zip(self._context.parents, parent_grads):
            if g.shape != p.data.shape:
                raise ValueError(f"Computed grad doesn't match data's shape. {g.shape} != {p.data.shape}.")
            p.grad.data += g

class Context:
    def __init__(self, func_applied, func_name, func_kwargs, *parents):
        self.func_applied = func_applied
        self.func_name = func_name
        self.parents = parents
        self.safe = []
        self.func_kwargs = func_kwargs

    def save_for_backward(self, *x):
        self.safe.extend(x)

class Function:
    def apply_func(self, func, func_name, *x, **kwargs):  # self is the Tensor that the function is applied on.
        context = Context(func, func_name, kwargs, self, *x)
        retval = Tensor(func.forward(context, self.data, *[t.data for t in x], **kwargs))
        retval._context = context
        return retval

# Load ops into Tensor class.
def register(name, func):
    setattr(Tensor, name, partialmethod(func.apply_func, func, name))

import minigrad.accelerators.acc_cpu

