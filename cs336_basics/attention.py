import torch


def softmax(x: torch.Tensor, dim: int):
    """Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution."""
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_x_sum
