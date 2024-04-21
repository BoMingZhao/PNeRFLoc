# PNeRFLoc [AAAI 24]

This is the official pytorch implementation of **PNeRFLoc: Visual Localization with Point-based Neural Radiance Fields**.

## Update

- [x] Code for Indoor Dataset (7Scenes, Replica)

- [ ] Code for Outdoor Dataset (Cambridge)

## Installation

We have tested the code on Python 3.8, 3.9 and PyTorch 1.8.1, 2.0.1 with CUDA 11.3 and 11.8, while a newer version of pytorch should also work. The steps of installation are as follows:

- create virtual environmental:   ``` conda create -n PNeRFLoc python=3.9 ```

- Activate the visual environment:  ``` conda activate PNeRFLoc ```

- Install dependencies:  ``` bash requirements.sh ```. The default installation is PyTorch 2.0.1 with CUDA 11.8. Please select the PyTorch and CUDA versions that are compatible with your GPU.

## Data Preparation

We use [7Scenes](https://pan.baidu.com/s/10Qacbk25fnTS-2chg0KJkA?pwd=myyq) Dataset and [Replica](https://drive.google.com/file/d/1VwxfkT4AlDWn5CbY6yvUH7KO7z4uhENz/view?usp=drive_link) Dataset. 

Then we need to extract R2D2 key points from the query image by running: 

```
bash dev_scripts/utils/generate_r2d2.sh
```

The layout should looks like this:

```
PNeRFLoc
├── data_src
│   ├── Replica
	│   │──room0
	│   │   │──exported
	│   │   │   │──color
    │   │   │   │──depth
    │   │   │   │──depth_render
    │   │   │   │──pose
    │   │   │   │──intrinsic
    │   │   │   │──r2d2_query
	│   │──room1
    ...
    ├── 7Scenes
    │   │──chess
	│   │   │──exported
	│   │   │   │──color
    │   │   │   │──depth
    │   │   │   │──depth_render
    │   │   │   │──pose
    │   │   │   │──intrinsic
    │   │   │   │──TrainSplit.txt
    │   │   │   │──TestSplit.txt
    │   │   │   │──r2d2_query
	│   │──pumpkin
	...
	│   │──7scenes_sfm_triangulated
```

## Train

Simply run

> bash ./dev_scripts/train/${Dataset}/${Scene}.sh

<details>
  <summary>Command Line Arguments for train</summary>
  <ul>
    <li><strong>scan</strong> 
    </li>
    Scene name.
  </ul>


  <ul>
    <li><strong>train_end</strong> 
    </li>
    Reference sequence cut-off ID.
  </ul>


  <ul>
    <li><strong>skip</strong> 
    </li>
    Select one image from every few images of the Reference sequence as the Training view.
  </ul>
  <ul>
    <li><strong>vox_res</strong> 
    </li>
    Resolution of voxel downsampling.
  </ul>


  <ul>
    <li><strong>gpu_ids</strong> 
    </li>
    GPU ID.
  </ul>

</details>

## Optimize

Simply run

> bash ./dev_scripts/loc/${Dataset}/${Scene}.sh

<details>
  <summary>Command Line Arguments for optimize</summary>
  <ul>
    <li><strong>format</strong> 
    </li>
    0 indicates optimizing the pose using quaternions, 1 indicates optimizing the pose using SE3, and 2 indicates optimizing the pose using a 6D representation.
  </ul>


  <ul>
    <li><strong>save_path</strong> 
    </li>
    Path to the optimized pose results. Note that if you change this path, please also pass the correct path through args when evaluating.
  </ul>


  <ul>
    <li><strong>per_epoch</strong> 
    </li>
    Number of optimizations per image, 250 by default.
  </ul>
  <ul>
    <li><strong>render_times</strong> 
    </li>
    Total number of renderings during optimization, 1 by default. Therefore, the total number of optimizations equals **_per_epoch_ * _render_times_**.
  </ul>


</details>

## Evaluation

Once you have optimized all scenes in a dataset, you can evaluate it like this:

```
python evaluate_${Dataset}.py
```

You can also evaluate specific scenes by manually entering the names of the scenes you want to evaluate.

```
python evaluate_${Dataset}.py --scene ${scene_name1} ${scene_name2} ... ${scene_nameN}
```

## Citing

```
@inproceedings{zhao2024pnerfloc,
  title={PNeRFLoc: Visual Localization with Point-based Neural Radiance Fields},
  author={Zhao, Boming and Yang, Luwei and Mao, Mao and Bao, Hujun and Cui, Zhaopeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7450--7459},
  year={2024}
}
```

