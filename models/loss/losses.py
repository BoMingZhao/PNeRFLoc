"""
Generic losses and error functions for optimization or training deep networks.
"""

import torch


def scaled_loss(x, fn, a):
    """Apply a loss function to a tensor and pre- and post-scale it.
    Args:
        x: the data tensor, should already be squared: `x = y**2`.
        fn: the loss function, with signature `fn(x) -> y`.
        a: the scale parameter.
    Returns:
        The value of the loss, and its first and second derivatives.
    """
    a2 = a**2
    loss, loss_d1, loss_d2 = fn(x/a2)
    return loss*a2, loss_d1, loss_d2/a2


def squared_loss(x):
    """A dummy squared loss."""
    return x, torch.ones_like(x), torch.zeros_like(x)


def huber_loss(x):
    """The classical robust Huber loss, with first and second derivatives."""
    mask = x <= 1
    sx = torch.sqrt(x)
    isx = torch.max(sx.new_tensor(torch.finfo(torch.float).eps), 1/sx)
    loss = torch.where(mask, x, 2*sx-1)
    loss_d1 = torch.where(mask, torch.ones_like(x), isx)
    loss_d2 = torch.where(mask, torch.zeros_like(x), -isx/(2*x))
    return loss, loss_d1, loss_d2


def barron_loss(x, alpha, derivatives: bool = True, eps: float = 1e-7):
    """Parameterized  & adaptive robust loss function.
    Described in:
        A General and Adaptive Robust Loss Function, Barron, CVPR 2019

    Contrary to the original implementation, assume the the input is already
    squared and scaled (basically scale=1). Computes the first derivative, but
    not the second (TODO if needed).
    """
    loss_two = x
    loss_zero = 2 * torch.log1p(torch.clamp(0.5*x, max=33e37))

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.abs(alpha - 2.).clamp(min=eps)
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(
        alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
    alpha_safe = alpha_safe * torch.abs(alpha).clamp(min=eps)

    loss_otherwise = 2 * (beta_safe / alpha_safe) * (
        torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == 0, loss_zero,
        torch.where(alpha == 2, loss_two, loss_otherwise))
    dummy = torch.zeros_like(x)

    if derivatives:
        loss_two_d1 = torch.ones_like(x)
        loss_zero_d1 = 2 / (x + 2)
        loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)
        loss_d1 = torch.where(
            alpha == 0, loss_zero_d1,
            torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

        return loss, loss_d1, dummy
    else:
        return loss, dummy, dummy


def scaled_barron(a, c):
    return lambda x: scaled_loss(
            x, lambda y: barron_loss(y, y.new_tensor(a)), c)

def weighted_mse(x):
    weights = torch.ones_like(x)
    return x, weights, weights

def mse():
    return lambda x: weighted_mse(x)
