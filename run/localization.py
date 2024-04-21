import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
import pathlib
import sys
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import torch
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
import time
from options import LocalizationOptions
from data import create_data_loader, create_dataset
from models import load_model
import copy
from utils.cam import transpose, pi, pi_inv, rotation_6d_to_matrix, log_map_so3, exp_map_so3
from data.data_utils import get_dtu_raydir_tensor
import torch.nn.functional as F
from typing import Tuple
import cv2
from tqdm import tqdm
from data.colmap_utils import qvec2rotmat_tensor, rotmat2qvec_tensor
from einops import repeat
from torch.autograd import Variable
from models.LM_utils.run import run

def mse2psnr(x): return -10.* torch.log(x) / np.log(10.)

def save_results(save_path, id, c2w):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{id:d}.txt')
    np.savetxt(save_path, c2w.tolist())

def test(model, data, opt, c2w):
    model.eval()
    patch_size = 90
    chunk_size = patch_size * patch_size
    data['c2w'] = c2w.view([1, 4, 4])
    data['camrotc2w'] = (c2w[0:3, 0:3]).view([1, 3, 3])
    data['campos'] = c2w[0:3, 3].view([1, 3])
    raydir = get_dtu_raydir_tensor(data["pixel_idx"].squeeze().to(c2w.device), 
                        data["intrinsic"].squeeze().to(c2w.device), c2w[0:3, 0:3], opt.dir_norm > 0).view([1, -1, 3])
    pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
    totalpixel = pixel_idx.shape[1]
    data.pop('gt_mask', None)
    visuals = None
    for k in range(0, totalpixel, chunk_size):
        start = k
        end = min([k + chunk_size, totalpixel])
        data['raydir'] = raydir[:, start:end, :]
        data["pixel_idx"] = pixel_idx[:, start:end, :]
        model.set_input(data)
        model.test()
        curr_visuals = model.get_current_visuals(data=data)
        if visuals is None:
            visuals = {}
            for key, value in curr_visuals.items():
                visuals[key] = value
        else:
            for key, value in curr_visuals.items():
                visuals[key] = torch.cat([visuals[key], value], dim=1)
    return visuals

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]

def get_cam_tensor(c2w, device='cuda', split=False, which_format=0):
    R = c2w[:3, :3]
    if which_format == 1: # Use so3
        quad = log_map_so3(R).view([3])
    elif which_format == 2: # Use 6D rotation representation by Zhou et al.
        quad = R[:2, :].reshape([6])
    else:  # Use Quaternion
        quad = rotmat2qvec_tensor(R).view([4])
    t = c2w[:3, 3].view([3])
    if split:
        T = Variable(t.detach().to(device), requires_grad=True)
        cam_para_list_T = [T]
        quad = Variable(quad.detach().to(device), requires_grad=True)
        cam_para_list_quad = [quad]
        return T, quad, cam_para_list_T, cam_para_list_quad
    camera_tensor = torch.cat([quad, t], dim=0).detach()
    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
    cam_para_list = [camera_tensor]
    return camera_tensor, cam_para_list

def from_cam_tensor_to_c2w(cam_tensor, bottom):
    if cam_tensor.shape[0] == 7: # Quaternion to matrix
        quad = cam_tensor[:4]
        R = qvec2rotmat_tensor(quad)
    elif cam_tensor.shape[0] == 9: # 6D to matrix
        rotation_6d = cam_tensor[:6]
        R = rotation_6d_to_matrix(rotation_6d)
    else: # so3 to matrix
        rvec = cam_tensor[:3]
        R = exp_map_so3(rvec)
    t = cam_tensor[-3:].view([3, 1])

    return torch.cat([torch.cat([R, t], dim=-1), bottom], dim=0)

def mask_in_image(pts, image_size: Tuple[int, int], pad: int = 1):
    w, h = image_size
    image_size_ = torch.tensor([w - pad - 1, h - pad - 1]).to(pts)
    return torch.all((pts >= pad) & (pts <= image_size_), -1)

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def main():
    opt = LocalizationOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                                   gpu_ids else torch.device('cpu'))
    bottom = torch.tensor([0., 0., 0., 1.]).view([1, 4]).to(device)
    print('----------------------device: ', device, '----------------------')
    print('----------------------Loading Data----------------------')
    with torch.no_grad():
        model = load_model(opt)
        point_feature_all = model.neural_points.points_embeding.squeeze()[:, 7:]
        conf = model.neural_points.points_embeding.squeeze()[:, 6]
        point_xyz_all = model.neural_points.xyz
        if hasattr(model.neural_points, 'scores'):
            print(f'Before score mask, there are {point_xyz_all.shape[0]:d} points')
            point_scores = model.neural_points.scores.squeeze()
            score_mask = (point_scores * conf) >= 0.9
            point_xyz_all = point_xyz_all[score_mask]
            point_feature_all = point_feature_all[score_mask]
            print(f'After score mask, there are {point_xyz_all.shape[0]:d} points')
        torch.cuda.empty_cache()
    P_N = point_feature_all.shape[0]

    print("# Create Optimize Datasets!")
    test_opt = copy.deepcopy(opt)
    test_opt.split = 'test'
    test_opt.batch_size = 1
    test_opt.prob = 0
    test_opt.test_num_step = opt.test_num_step
    test_dataset = create_dataset(test_opt)

    data_loader = create_data_loader(test_opt, dataset=test_dataset)
    model.opt.is_train = 0
    model.opt.no_loss = 1
    dataset_size = len(data_loader)

    print('# optimize images = {}'.format(dataset_size))
    height = test_dataset.height
    width = test_dataset.width

    start_time = time.time()
    is_outdoor = True if test_opt.dataset_name == "cambridge" else False
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            test_dataset.rand = 0
            id = batch['vid'].item()
            K = batch['intrinsic'].squeeze().numpy().astype(np.float32)

            keypoints = batch['r2d2_keypoints'].to(device).squeeze()
            image_feature = batch['r2d2_feature'].to(device).squeeze()
            f_N = image_feature.shape[0]
            similarity = []
            for part in range(0, P_N, test_opt.chunk):
                x = repeat(point_feature_all[part: part + test_opt.chunk, :], 'n1 n2 -> n1 n3 n2', n3=f_N)
            
                y = repeat(image_feature, 'n1 n2 -> n3 n1 n2', n3=x.shape[0])
                similarity.append(torch.cosine_similarity(x, y, dim=-1))
            similarity = torch.cat(similarity, dim=0)
            _, similarity_index = torch.max(similarity, dim=0)

            point_vis = point_xyz_all[similarity_index].cpu().numpy().astype(np.double)
            keypoints_numpy = keypoints[..., :2].cpu().numpy().astype(np.double)
            _, R, t, inliers = cv2.solvePnPRansac(point_vis, keypoints_numpy, K, distCoeffs=None, flags=cv2.SOLVEPNP_DLS, 
                                   iterationsCount=20000, reprojectionError=test_opt.inliers_thres)
            if inliers is not None:
                if inliers.size > 4:
                    R, t = cv2.solvePnPRefineLM(point_vis[inliers], keypoints_numpy[inliers], 
                                                K, distCoeffs=None, rvec=R, tvec=t)
            R, _ = cv2.Rodrigues(R)
            
        pixelcoords = batch['pixel_idx'].view([1, -1, 2]).to(device)
        w2c = torch.cat([torch.from_numpy(np.concatenate([R, t], axis=-1)).to(device), bottom], dim=0).to(torch.float32)
        c2w = torch.inverse(w2c)

        if test_opt.pnp_status != 0:
            save_results(os.path.join(test_opt.save_path, "pnp_pose"), id, c2w.detach().cpu().numpy())
            if test_opt.pnp_status == 1:
                continue
        K = batch['intrinsic'].squeeze().to(torch.float32).to(device)
        with torch.no_grad():
            v = test(model, batch, test_opt, c2w=c2w)
            coarse_raycolor = v['coarse_raycolor'].view([-1, 3]) # Debug
            depth = v['coarse_depth'].view([-1, 1])
            coarse_raycolor = torch.clip(coarse_raycolor, 0., 1.)
        eps = 0.001 * torch.ones_like(depth)
        depth = torch.where(depth == 0.0, eps, depth)
        depth_mask = depth[:, 0] > 0.01
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3].view([3])
        point_3d_world = transpose(R_c2w, t_c2w, pi_inv(K, pixelcoords, depth))

        gt_image = batch['images'].to(device).squeeze(0)

        if test_opt.optimize_method == 0:
            c2w_save = run(batch["h"].item(), batch["w"].item(), c2w, K, bottom, depth_mask, point_3d_world, coarse_raycolor, gt_image)
        else:
            if is_outdoor:
                gt_mask = batch['outdoor_mask'].to(device).permute(0, 3, 1, 2)

            cam_tensor, cam_para_list = get_cam_tensor(c2w.cpu(), which_format=opt.format)
            print('----------------------set optimizer----------------------')
            optimizer = torch.optim.Adam(cam_para_list, lr=opt.lr)

            all_epoch = test_opt.per_epoch * test_opt.render_times
            loop = tqdm(range(all_epoch))
            for epoch in loop:
                if epoch != 0 and epoch % test_opt.per_epoch == 0:
                    c2w = c2w_render.clone()
                    print('----------------------Rerender----------------------')
                    cam_tensor, cam_para_list = get_cam_tensor(c2w.cpu())
                    optimizer = torch.optim.Adam(cam_para_list, lr=opt.lr)
                    multiStep_data = test_dataset.get_item(batch_idx)
                    pixelcoords = multiStep_data['pixel_idx'].view([1, -1, 2]).to(device)
                    with torch.no_grad():
                        v = test(model, multiStep_data, test_opt, c2w=c2w)
                        coarse_raycolor = v['coarse_raycolor'].view([-1, 3])
                        depth = v['coarse_depth'].view([-1, 1])
                    eps = 0.001 * torch.ones_like(depth)
                    depth = torch.where(depth == 0.0, eps, depth)
                    depth_mask = depth[:, 0] > 0.01
                    R_c2w = c2w[:3, :3]
                    t_c2w = c2w[:3, 3].view([3])
                    point_3d_world = transpose(R_c2w, t_c2w, pi_inv(K, pixelcoords, depth))

                c2w = from_cam_tensor_to_c2w(cam_tensor, bottom)
                R = c2w[:3, :3].t()
                tc2w = c2w[:3, 3].view([3])
                t = - R @ tc2w

                point_3d_cam = transpose(R, t, point_3d_world)
                point_2d, _ = pi(K, point_3d_cam)
                mask = mask_in_image(point_2d, (width, height), 0) & depth_mask
                point_2d = point_2d[mask, ...]
                if point_2d.shape[0] <= 10:
                    tqdm.write("Too few valid points. Save PnP pose! ")
                    if epoch == 0:
                        c2w_save = c2w.detach().cpu().numpy()
                    break
                point_2d[..., 0] = 2 * (point_2d[..., 0] / (width - 1)) - 1
                point_2d[..., 1] = 2 * (point_2d[..., 1] / (height - 1)) - 1
                color_gt_sample = F.grid_sample(gt_image, point_2d.view([1, 1, -1, 2]), align_corners=True).squeeze()
                if is_outdoor:
                    color_gt_mask = F.grid_sample(gt_mask, point_2d.view([1, 1, -1, 2]), align_corners=True, mode='nearest').squeeze().to(torch.bool)
                color_gt_sample = color_gt_sample.permute(1, 0)
                if is_outdoor:
                    loss_warp = torch.nn.MSELoss()(color_gt_sample[color_gt_mask], coarse_raycolor[mask, ...][color_gt_mask])
                else:
                    loss_warp = torch.nn.MSELoss()(color_gt_sample, coarse_raycolor[mask, ...])
                if torch.isnan(loss_warp):
                    tqdm.write("Sorry, the loss has nan in it!")
                    break

                postfix = {'epoch': epoch}
                postfix['loss'] = loss_warp.item()
                loop.set_postfix(postfix)

                if epoch == 0 or epoch % test_opt.per_epoch == 0:
                    best_loss = postfix['loss']

                try:
                    loss_warp.backward()
                except:
                    tqdm.write("Optimize failed. Save PnP pose!")
                    if epoch == 0:
                        c2w_save = c2w.detach().cpu().numpy()
                    break
                optimizer.step()
                optimizer.zero_grad()
                
                if best_loss > postfix['loss']:
                    best_loss = postfix['loss']
                    c2w_save = c2w.detach().cpu().numpy()
                    c2w_render = c2w.detach()
        save_results(test_opt.save_path, id, c2w_save)
    print(f"# Save the results at: {test_opt.save_path}")
    print('# Average Time Consumption:', (time.time() - start_time) / len(test_dataset))


if __name__ == '__main__':
    main()
