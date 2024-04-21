import numpy as np
import os
import torch
from argparse import ArgumentParser
from prettytable import PrettyTable
from data.colmap_utils import read_images_binary

def transform_scannet_to_colmap_id(scan, data_dir, step=1000):
    transform_id = {}
    seq_scannet_list = []
    with open(os.path.join(data_dir, scan, "exported", 'TrainSplit.txt')) as f:
        for line in f:
            seq = int(line[8:])
            seq_scannet_list.append(seq)
    with open(os.path.join(data_dir, scan, "exported", 'TestSplit.txt')) as f:
        for line in f:
            seq = int(line[8:])
            seq_scannet_list.append(seq)
    seq_colmap_list = sorted(seq_scannet_list)
    for i in range(len(seq_scannet_list) * step): # id in scannet
        seq = seq_scannet_list[int(i / step)]
        frame = i % step
        index = seq_colmap_list.index(seq) * step + 1 + frame
        transform_id[i] = index
    return transform_id

def from_id_to_pose(colmap_id, colmap_data):
    bottom = np.array([0., 0., 0., 1.]).reshape([1, 4]).astype(np.float32)
    tw2c = colmap_data[colmap_id].tvec
    Rw2c = colmap_data[colmap_id].qvec2rotmat()
    R = Rw2c.T
    t = -R @ tw2c
    c2w = np.concatenate([np.concatenate([R, t.reshape([3, 1])], axis=-1), bottom], axis=0).astype(np.float32)
    return c2w

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
    TrainEnd_Length_dict = {'chess': (4000, 2000), 'fire': (2000, 2000), 'heads': (1000, 1000), 
                            'redkitchen': (7000, 5000), 'office': (6000, 4000), 'pumpkin': (4000, 2000),
                            'stairs': (2000, 1000)}
    scenes = ["chess", "pumpkin", "office", "stairs", "heads", "fire", "redkitchen"]
    
    parser = ArgumentParser(description="Evaluate parameters")
    parser.add_argument('--gt_path', '-g', type=str, default="./data_src/7Scenes/")
    parser.add_argument('--optimize_path', '-op', type=str, default="./results/7Scenes/")
    parser.add_argument('--optimize_name', '-on', type=str, default="pose")
    parser.add_argument('--scene', nargs='*')
    args = parser.parse_args()

    if args.scene is not None:
        scenes = args.scene
    info = []
    for scene in scenes:
        colmap_data = read_images_binary(os.path.join('./data_src/7Scenes/7scenes_sfm_triangulated/', scene, 'triangulated', 'images.bin'))
        step = 500 if scene == 'stairs' else 1000
        transform_id = transform_scannet_to_colmap_id(scene, args.gt_path, step)
        optimize_path = os.path.join(args.optimize_path, scene, args.optimize_name)
        error_t_optimize, error_R_optimize = [], [] 
        recall_cnt = 0
        start, length = TrainEnd_Length_dict[scene]
        for i in range(start, start + length):
            pose_gt = from_id_to_pose(transform_id[i], colmap_data)
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