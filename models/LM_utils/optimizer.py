from packaging import version
import torch
from torch import Tensor

if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky

def optimizer_step(g, H, lambda_=0, mute=False, mask=None, eps=1e-6, stable_lambda=False):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
        mask: denotes valid elements of the batch (optional).
    """
    if lambda_ is 0:  # noqa
        diag = torch.zeros_like(g)
    elif stable_lambda:
        diag = lambda_ * torch.ones_like(g)
    else:
        diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
    H = H + diag.clamp(min=eps).diag_embed()

    if mask is not None:
        # make sure that masked elements are not singular
        H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
        # set g to 0 to delta is 0 for masked elements
        g = g.masked_fill(~mask[..., None], 0.)

    H_, g_ = H.cpu(), g.cpu()
    try:
        U = cholesky(H_)
    except RuntimeError as e:
        if 'singular U' in str(e):
            if not mute:
                print(
                    'Cholesky decomposition failed, fallback to LU.')
            delta = -torch.solve(g_[..., None], H_)[0][..., 0]
        else:
            raise
    else:
        delta = -torch.cholesky_solve(g_[..., None], U)[..., 0]

    return delta.to(H.device)

def build_system(J: Tensor, res: Tensor, weights: Tensor):
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weights[..., None] * grad
        grad = grad.sum(-2)  # ... x 6

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3)  # ... x 6 x6

        return grad, Hess

def early_stop(**args):
    stop = False
    if (args['i'] % 10) == 0:
        T_delta, grad = args['T_delta'], args['grad']
        grad_norm = torch.norm(grad.detach(), dim=-1)
        small_grad = grad_norm < 0.00001
        dR, dt = T_delta.magnitude()
        small_step = ((dt < 0.00001)
                        & (dR < 0.0001))
        if torch.all(small_step | small_grad):
            stop = True
    return stop