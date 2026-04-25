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
        server_port=8098,
        model_name='cma_clip',
        ckpt_path='checkpoints/20251109_vlnverse_cma_clip/ckpts/checkpoint-610240',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='vln_pe',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='20251113_cma_clip_flash_vlnverseFineTrain_ckpt610240',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 500,
            'warm_up_step': 500
        },
        scene=SceneCfg(
            scene_type='kujiale',
            scene_data_dir='data/scene_data/vlnverse',
        ),
        robot_name='h1',
        robot_flash=True,  # If robot_flash is True, the mode is flash (set world_pose directly); else you choose physical mode.
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_type="kujiale",
        dataset_settings={
            # 'base_data_dir': 'data/vlnverse/raw_data/vlnverse/mixed_splits',
            'base_data_dir': 'data/vlnverse/raw_data/vlnverse/final_splits_with_distance_formal/fine',
            'split_data_types': ['train'],
            # 'split_data_types': ['test_w61'],
            'filter_stairs': False,
        },
    ),
)
