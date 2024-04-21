#!/bin/bash
nrCheckpoint="../checkpoints/"
nrDataRoot="../data_src/"
name='heads_vox500_renderdepth_ColmapPose'
train_end=1000
lr=0.001
n_threads=4
r2d2_feature=1
rendering_only=0

no_loss=1
compute_depth=1
format=2
pnp_status=2
scan="heads"
save_path="../results/7Scenes/${scan}/pose/"
serial_batches=1

resume_iter=30000 # 20000
data_root="${nrDataRoot}/7Scenes/"
edge_filter=50

point_conf_mode="01" # 0 for only at features, 1 for multi at weight
point_dir_mode="01" # 0 for only at features, 1 for color branch
point_color_mode="01" # 0 for only at features, 1 for color branch

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=4
depth_limit_scale=0
alpha_range=0

vscale=" 2 2 2 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
vsize=" 0.008 0.008 0.008 " #" 0.005 0.005 0.005 "
wcoord_query=-1
z_depth_dim=400
max_o=200000
ranges=" -8.0 -8.0 -8.0 8.0 8.0 8.0 "
SR=24
K=8
P=26
NN=2

act_type="LeakyReLU"

agg_intrp_order=2
agg_distance_kernel="linear" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8

point_features_dim=135
shpnt_jitter="uniform" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=0 #2
shading_feature_mlp_layer1=4 #2
shading_feature_mlp_layer2=0 #1
shading_feature_mlp_layer3=4 #1
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0


raydist_mode_unit=1
dataset_name='scenes7_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=0.1
far_plane=7
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'

random_sample_size=300 #48 # 32 * 32 = 1024
batch_size=1

gpu_ids='0'

checkpoints_dir="${nrCheckpoint}/7scenes/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_color_dir_agg2_128_perpixel"

test_num_step=20
visual_items=' coarse_raycolor gt_image coarse_depth'
color_loss_weights=" 1.0 0.0 0.0 "
color_loss_items='ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor'
test_color_loss_items='coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor'

bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

cd run

python3 localization.py \
        --format $format \
        --name $name \
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --n_threads $n_threads \
        --rendering_only $rendering_only \
        --no_loss $no_loss \
        --save_path $save_path \
        --pnp_status $pnp_status \
        --compute_depth $compute_depth \
        --r2d2_feature $r2d2_feature \
        --serial_batches $serial_batches \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --pin_data_in_memory $pin_data_in_memory \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --bg_color $bg_color \
        --split $split \
        --train_end $train_end \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --edge_filter $edge_filter \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --lr $lr \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --visual_items $visual_items \
        --act_type $act_type \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --alpha_range $alpha_range \
        --ranges $ranges \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --max_o $max_o \
        --debug

