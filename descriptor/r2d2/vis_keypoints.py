import os
import numpy as np
from PIL import Image
import imageio
from matplotlib import pyplot as plt
import torch
import cv2

def generate_mask(depth_path, output_path):
    depth = torch.from_numpy(np.array(Image.open(depth_path).convert('RGB')))
    depth_sum = torch.sum(depth, dim=-1, keepdim=True)
    print(depth_sum.shape)
    mask = depth_sum != 0.
    mask = torch.masked_fill(depth, mask, 255).numpy()
    imageio.imwrite(output_path, mask)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Visualize the keypoints")
    
    parser.add_argument("--img", type=str, default="imgs/r_0.png")
    parser.add_argument("--np_file", type=str, default="imgs/r_0.png.r2d2")

    args = parser.parse_args()

    img = Image.open(args.img).convert('RGB')
    img = np.array(img)
    with open(args.np_file, 'rb') as f:
        data = np.load(f)
        kpts = data['keypoints']
        scores = data['scores']

    print(scores.max(), scores.min())
    print(kpts.shape)
    plt.imshow(img)
    plt.scatter(kpts[:, 0], kpts[:, 1])
    plt.show()

    img = cv2.imread(args.img)
    with open(args.np_file, 'rb') as f:
        data = np.load(f)
        kpts = data['keypoints']
        scores = data['scores']
    out = img
    for i in range(kpts.shape[0]):
        x1 = kpts[i][0]
        y1 = kpts[i][1]
        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 3, (0, 0, 255), -1)
    cv2.imwrite('viz.png', out)
        