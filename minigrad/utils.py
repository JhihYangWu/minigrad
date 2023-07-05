def calc_nc_grad(tensors, loss_tensor):
    # Calculates gradients numerically for tensors with respect to loss_tensor.
    loss_tensor.backward()

