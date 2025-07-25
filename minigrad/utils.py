from minigrad.tensor import Tensor
import numpy as np

def calc_nc_grad(tensors, loss_tensor, eps=1e-6):
    def append_tensor(eval_str, t, i, j, to_append, comp_path, prev_str):
        if to_append is t:
            eval_str.append("t")
        else:
            if i+1 < len(comp_path) and to_append is comp_path[i+1][2]:
                eval_str.append(prev_str)
            else:
                eval_str.append(f"comp_path[{i}][0][{j}]")
    # Calculates gradients numerically for tensors with respect to loss_tensor.
    for t in tensors:
        # Do DFS to find tensor.
        comp_path = []
        found = _nc_grad_helper(loss_tensor, t, comp_path)
        assert found and comp_path != []

        # Change all the tensors in comp_path to float64 for better precision.
        for row in comp_path:
            for ts in row[0]:
                if ts.data.dtype != np.float64:
                    ts.data = ts.data.astype(np.float64)
            if row[2].data.dtype != np.float64:
                row[2].data = row[2].data.astype(np.float64)

        # Find eval string to compute loss from t.
        i = len(comp_path) - 1
        prev_str = None
        while i >= 0:
            eval_str = []
            for j, x in enumerate(comp_path[i][0]):
                if j == 1:
                    eval_str.extend((".", comp_path[i][1], "("))
                if j >= 2:
                    eval_str.append(", ")
                append_tensor(eval_str, t, i, j, x, comp_path, prev_str)
            unary = len(comp_path[i][0]) == 1
            if unary:
                # Unary ops never got chance to add op name to eval_str.
                eval_str.extend((".", comp_path[i][1], "("))
            for index, key in enumerate(comp_path[i][3]):
                if not unary or index >= 1:
                    eval_str.append(", ")
                eval_str.append(f"{key}={comp_path[i][3][key]}")
            eval_str.append(")")
            eval_str = "".join(eval_str)
            prev_str = eval_str
            i -= 1
        complete_eval_str = prev_str

        # Find grads numerically.
        t.grad = Tensor(np.zeros(t.shape, dtype=np.float32))
        for index in find_all_indices(t.shape):
            old_val = t.data[index]
            t.data[index] = old_val + eps
            r_loss = eval(complete_eval_str).data
            t.data[index] = old_val - eps
            l_loss = eval(complete_eval_str).data
            t.data[index] = old_val
            t.grad.data[index] = (r_loss - l_loss) / (2 * eps)

def _nc_grad_helper(t, target, comp_path):
    if t is target:
        return True
    if t._context is None:
        return False

    comp_path.append((t._context.parents, t._context.func_name, t, t._context.func_kwargs))
    found = False
    for p in t._context.parents:
        found = found or _nc_grad_helper(p, target, comp_path)
        if found:
            return True
    comp_path.pop()
    return False

def find_all_indices(shape):
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    retval = []
    p = find_all_indices(shape[1:])
    for i in range(shape[0]):
        for j in p:
            retval.append((i,) + j)
    return retval

def get_mnist():
    def fetch(url):
        import requests, gzip, os, hashlib, tempfile
        fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                data = f.read() 
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    x_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
    x_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    y_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    x_train = x_train.reshape((-1, 784)).astype(np.float32)
    x_test = x_test.reshape((-1, 784)).astype(np.float32)
    y = np.zeros((len(y_train), 10), dtype=np.float32)
    y[range(y.shape[0]), y_train] = 1
    y_train = y
    y = np.zeros((len(y_test), 10), dtype=np.float32)
    y[range(y.shape[0]), y_test] = 1
    y_test = y

    return x_train, y_train, x_test, y_test

