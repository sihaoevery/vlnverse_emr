from vlnverse.configs.agent import AgentCfg
from vlnverse.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8084,
        model_name='rdp',
        ckpt_path='checkpoints/20251109_rdp_vlnverse_c2/ckpts/checkpoint-82001',
        model_settings={
            # debug
            'vis_debug': False,  # If vis_debug=True, you can get visualization results
            'vis_debug_path': './logs/20251208_rdp_flash_vlnverseRealFine_ckptC2_82001_train_flash_collision/vis_debug',
        },
    ),
    env=EnvCfg(
        env_type='vln_pe',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='20251208_rdp_vlnverse_ckptC2_82001_flash_collision_noLight_env2_proc2_fine',
        task_settings={
            'env_num': 2,
            'use_distributed': False,
            'proc_num': 2,
            'max_step': 500 # for flash mode.
        },
        scene=SceneCfg(
            scene_type='kujiale_no_light', # kujiale
            scene_data_dir='data/scene_data/vlnverse',
        ),
        robot_name='h1',
        robot_flash=True,  # If robot_flash is True, the mode is flash (set world_pose directly); else you choose physical mode.
        flash_collision=True,  # If flash_collision is True, the robot will stop when collision detected.
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud_vlnverse.usd',
        camera_resolution=[256, 256],  # (W,H)
        # vis_output_resolution=[640, 480],
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_type="kujiale",
        dataset_settings={
            # 'base_data_dir': 'data/vln_pe/raw_data/vlnverse/mixed_splits',
            'base_data_dir': 'data/vln_pe/raw_data/vlnverse/final_splits_with_distance_formal/fine',
            'split_data_types': ['val_seen', 'val_unseen', 'test'],
            # 'split_data_types': ['test_w61'],
            'filter_stairs': False,
        },
    ),
    eval_settings={
        'save_to_json': False, 
        'vis_output': False,  # save result to video under logs/
    }, 
)
