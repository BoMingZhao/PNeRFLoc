import torch
import numpy as np
from models.LM_utils.interpolation import Interpolator
from models.LM_utils.warppers import Pose, Camera
from models.LM_utils.costs import DirectAbsoluteCost
from models.loss.losses import mse
from models.LM_utils.optimizer import optimizer_step, build_system, early_stop

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def update_lambda(lambda_, accept, method='traditional', mult=10, gain_factor=1, max=1e10, min=1e-10, eps1=0.25, eps2=0.75):
    if method == 'traditional':
        lambda_ = lambda_ * torch.where(accept, 1/mult, mult)
    elif method == 'Marquart':
        if gain_factor > eps2:
            lambda_ /= 3
        elif gain_factor < eps1:
            lambda_ *= 2
    else:
        lambda_ = lambda_ * torch.where(accept, mult / mult, mult)
        mult = torch.where(accept, 2 * (mult / mult), mult * 2)
    return lambda_.clamp(max=max, min=min), mult

def run(height, width, c2w, K, bottom, depth_mask, points_3d, render_color, gt_image):
    interpolator = Interpolator(mode='linear', pad=0)
    cost_fn = DirectAbsoluteCost(interpolator)
    loss_fn = mse()
    stable_lambda = False

    R = c2w[:3, :3].t()
    tc2w = c2w[:3, 3].view([3])
    t = - R @ tc2w
    T_init = Pose.from_Rt(R, t)
    T = T_init

    # masked blank depth area
    camera = Camera.from_para(width, height, K.detach()).to(torch.float32).to('cuda')
    F_ref = render_color[depth_mask]
    p3D = points_3d[depth_mask]
    F_query = gt_image.squeeze(0)
    args = (camera, p3D, F_ref, F_query)

    # set dampling factor
    failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)
    lambda_ = torch.full_like(failed, 0.001, dtype=T.dtype)
    mult = torch.full_like(lambda_, 10)
    recompute = True

    with torch.no_grad():
        res, valid_i, _ = cost_fn.residuals(T_init, *args)[:3]
        cost_i = loss_fn((res.detach()**2).sum(-1))[0]
        cost_best = masked_mean(cost_i, valid_i, -1)

        for i_lm in range(100):
            if recompute:
                res, valid, _, _, J = cost_fn.residual_jacobian(T, *args)
                failed = failed | (valid.long().sum(-1) < 10)
                cost = (res**2).sum(-1)
                cost, w_loss, _ = loss_fn(cost)
                weights = w_loss * valid.float()
                g, H = build_system(J, res, weights)
                # compute lambda if use stable lambda
                if i_lm == 0 and stable_lambda:
                    lambda_ *= H.diagonal(dim1=-2, dim2=-1).mean()

            delta = optimizer_step(g, H, lambda_, mask=~failed, stable_lambda=stable_lambda)
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T_new = T_delta @ T

            R_new = T_new.R.cpu().numpy()
            U, _, Vt = np.linalg.svd(R_new, full_matrices=True)
            R_new = torch.from_numpy(np.dot(U, Vt)).to('cuda')
            T_new = Pose.from_Rt(R_new, T_new.t)
            
            res, valid_accept = cost_fn.residuals(T_new, *args)[:2]
            cost_new = loss_fn((res**2).sum(-1))[0]
            cost_new = masked_mean(cost_new, valid_accept, -1)
            
            diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
            gain_factor = 2 * (cost_best - cost_new) / (torch.dot(torch.mul(delta, diag), delta) - torch.dot(g, delta))
            gain_factor = cost_best - cost_new
            accept = gain_factor > 0
            lambda_, mult = update_lambda(lambda_, accept, method='traditional', mult=mult)
            T = Pose(torch.where(accept[..., None], T_new._data, T._data))
            cost_best = torch.where(accept, cost_new, cost_best)
            recompute = accept.any()

            if early_stop(i=i_lm, T_delta=T_delta, grad=g, cost=cost):
                break

    if failed.any():
        print('One batch element had too few valid points.')
        pose = T_init.inv()
    else:
        pose = T.inv()
    R = pose.R.cpu().numpy()
    t = pose.t.view([3, 1]).cpu().numpy()
    c2w_save = np.concatenate([np.concatenate([R, t], axis=-1), bottom.cpu().numpy()], axis=0)

    return c2w_save