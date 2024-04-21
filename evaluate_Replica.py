import numpy as np
import os
import torch
from argparse import ArgumentParser
from prettytable import PrettyTable

def evaluate(c2w_gt, c2w, do_svd=False):
    if isinstance(c2w, torch.Tensor):
        if str(c2w.device) == 'cpu':
            c2w = c2w.detach().numpy()
        else:
            c2w = c2w.detach().cpu().numpy()
    if isinstance(c2w_gt, torch.Tensor):
        if str(c2w_gt.device) == 'cpu':
            c2w_gt = c2w_gt.detach().numpy()
        else:
            c2w_gt = c2w_gt.detach().cpu().numpy()
    
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    R_gt = c2w_gt[:3, :3]
    t_gt = c2w_gt[:3, 3]

    if do_svd:
        U, _, Vt = np.linalg.svd(R, full_matrices=True)
        R = np.dot(U, Vt)

    e_t = np.linalg.norm(t_gt - t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))

    return e_t, e_R

if __name__ == '__main__':
    scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    parser = ArgumentParser(description="Evaluate parameters")
    parser.add_argument('--gt_path', '-g', type=str, default="./data_src/Replica/")
    parser.add_argument('--optimize_path', '-op', type=str, default="./results/Replica/")
    parser.add_argument('--optimize_name', '-on', type=str, default="pose")
    parser.add_argument('--scene', nargs='*')
    args = parser.parse_args()

    if args.scene is not None:
        scenes = args.scene
    info = []
    for scene in scenes:
        gt_path = os.path.join(args.gt_path, scene, 'exported/pose')
        optimize_path = os.path.join(args.optimize_path, scene, args.optimize_name)
        error_t_optimize, error_R_optimize = [], [] 
        recall_cnt = 0
        length = len(os.listdir(os.path.join(optimize_path, "pnp_pose")))
        for i in range(2000, 2000 + length):
            pose_gt = np.loadtxt(os.path.join(gt_path, f'{i:d}.txt')).astype(np.float32)
            pose_optimize = np.loadtxt(os.path.join(optimize_path, f'{i:d}.txt')).astype(np.float32)
            if np.any(np.isnan(pose_optimize)):
                pose_optimize = np.eye(4).astype(np.float32)
            e_t, e_R = evaluate(c2w_gt=pose_gt, c2w=pose_optimize)
            if e_t <= 0.05 and e_R <= 5:
                recall_cnt += 1 
                
            error_t_optimize.append(e_t)
            error_R_optimize.append(e_R)

        error_t_optimize, error_R_optimize = np.array(error_t_optimize), np.array(error_R_optimize)
        med_t_optimize, med_R_optimize = np.median(error_t_optimize), np.median(error_R_optimize)
        recall = recall_cnt * 100 / length
        info.append({"name": scene, "recall": f"{recall:.2f}%", "e_t": f"{med_t_optimize:.4f}", "e_r": f"{med_R_optimize:.4f}"})
    table = PrettyTable()
    table.field_names = ["Scene", "Median-Translation-Error", "Median-Rotation-Error", "Recall"]

    for data in info:
        table.add_row([data["name"], data["e_t"], data["e_r"], data["recall"]])

    print(table)