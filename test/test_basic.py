import unittest
import torch
import numpy as np
from minigrad.tensor import Tensor
from minigrad.utils import calc_nc_grad

Tensor.NEED_PRECISION = True

class TestBasic(unittest.TestCase):
    def test_grads_1(self):
        x_init = np.random.randn(1, 3).astype(np.float32)
        W_init = np.random.randn(3, 3).astype(np.float32)
        m_init = np.random.randn(1, 3).astype(np.float32)

        def get_minigrad():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            loss = x.matmul(W).relu()
            loss = loss.exp2().sqrt().sin()
            loss = loss.mul(m).add(m).softmax(dim=1).log2().sum()
            loss.backward()
            return loss.data, x.grad.data, W.grad.data

        def get_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            loss = x.matmul(W).relu()
            loss = loss.exp2().sqrt().sin()
            loss = loss.mul(m).add(m).softmax(dim=1).log2().sum()
            loss.backward()
            return loss.detach().numpy(), x.grad, W.grad

        def get_nc_grad():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            loss = x.matmul(W).relu()
            loss = loss.exp2().sqrt().sin()
            loss = loss.mul(m).add(m).softmax(dim=1).log2().sum()
            calc_nc_grad([x, W], loss)
            return loss.data, x.grad.data, W.grad.data

        minigrad_output = get_minigrad()

        for x, y in zip(minigrad_output, get_pytorch()):
            np.testing.assert_allclose(x, y, rtol=1e-4)

        for x, y in zip(minigrad_output, get_nc_grad()):
            np.testing.assert_allclose(x, y, rtol=1e-4)

    def test_grads_2(self):
        x_init = np.random.randn(1000, 625).astype(np.float64)
        W_init = np.random.randn(625, 80).astype(np.float64)
        c_init = np.random.randn(1000, 625).astype(np.float64)

        def get_minigrad():
            x = Tensor(x_init)
            W = Tensor(W_init)
            c = Tensor(c_init)
            loss = x.sub(c).div(c).matmul(W).exp2().reshape(shape=(-1, 50)).max().sqrt()
            loss.backward()
            return loss.data, x.grad.data, W.grad.data, c.grad.data

        def get_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            c = torch.tensor(c_init, requires_grad=True)
            loss = x.sub(c).div(c).matmul(W).exp2().reshape((-1, 50)).max().sqrt()
            loss.backward()
            return loss.detach().numpy(), x.grad, W.grad, c.grad

        for x, y in zip(get_minigrad(), get_pytorch()):
            np.testing.assert_allclose(x, y, rtol=1e-4)

    def test_grad_3(self):
        for stride in [1, 2, 3]:
            x_init = np.random.randn(5, 2, 10, 7).astype(np.float32)
            w_init = np.random.randn(4, 2, 3, 2).astype(np.float32)

            def get_minigrad():
                x = Tensor(x_init)
                w = Tensor(w_init)
                loss = x.conv2d(w, stride=stride).sum()
                loss.backward()
                return loss.data, x.grad.data, w.grad.data

            def get_pytorch():
                x = torch.tensor(x_init, requires_grad=True)
                w = torch.tensor(w_init, requires_grad=True)
                loss = torch.nn.functional.conv2d(x, w, stride=stride).sum()
                loss.backward()
                return loss.detach().numpy(), x.grad, w.grad

            for x, y in zip(get_minigrad(), get_pytorch()):
                np.testing.assert_allclose(x, y, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()

