import torch


def count_params(model: torch.nn.Module, verbose=False) -> int:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def common_broadcast(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = common_broadcast(x, y)
    return x + y


def batch_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = common_broadcast(x, y)
    return x * y


def batch_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = common_broadcast(x, y)
    return x - y


def batch_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = common_broadcast(x, y)
    return x / y
