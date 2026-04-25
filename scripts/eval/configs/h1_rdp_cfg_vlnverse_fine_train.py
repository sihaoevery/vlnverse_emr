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
        server_port=8080,
        model_name='rdp',
        ckpt_path='checkpoints/20251109_rdp_vlnverse_c2/ckpts/checkpoint-82001',
        model_settings={
            # debug
            'vis_debug': True,  # If vis_debug=True, you can get visualization results
            'vis_debug_path': './logs/20251113_rdp_flash_vlnverseTrainFine_ckptC2_82001/vis_debug',
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
        task_name='20251113_rdp_flash_vlnverseTrainFine_ckptC2_82001',
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
            # 'base_data_dir': 'data/vln_pe/raw_data/vlnverse/mixed_splits',
            'base_data_dir': 'data/vln_pe/raw_data/vlnverse/final_splits_with_distance_formal/fine',
            'split_data_types': ['train'],
            # 'split_data_types': ['test_w61'],
            'filter_stairs': False,
        },
    ),
)
