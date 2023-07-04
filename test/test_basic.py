import unittest
import torch
import numpy as np
from minigrad.tensor import Tensor

x_init = np.random.randn(1, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)

class TestBasic(unittest.TestCase):
    def test_grads(self):
        def get_minigrad():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            loss = x.matmul(W).relu()
            loss = loss.mul(m).add(m).sum()
            loss.backward()
            return loss.data, x.grad.data, W.grad.data

        def get_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            loss = x.matmul(W).relu()
            loss = loss.mul(m).add(m).sum()
            loss.backward()
            return loss.detach().numpy(), x.grad, W.grad

        for x, y in zip(get_minigrad(), get_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

if __name__ == "__main__":
    unittest.main()

