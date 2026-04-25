import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH)

import json
import logging

import numpy as np
import torch
from vln.src.models.init_policy import initialize_policy
from vln.src.models.LongCLIP.model import longclip
from vln.src.models.utils.bert_token import BertTokenizer
from vln.src.models.utils.feature_extract import extract_instruction_tokens
from vln.src.utils.glove_embedding import InstructionEmbedding
from vln.src.utils.logger import MyLogger
from vln.src.utils.utils import batch_obs
from vln.src.v2.envs.env_factory import get_eval_config
from vln_real_deploy.policy.utils import get_obs


class PR_CMA_Agent:
    def __init__(self, project_path, logger=None, model_type='cma_clip', pe_finetuned=True):
        self.device = torch.device("cuda:0")
        self.pe_finetuned = pe_finetuned
        # 初始化config
        self._get_config(project_path, model_type, pe_finetuned)

        # 初始化logger
        self.logger = MyLogger(
            name="train",
            level=logging.INFO,
            format_str="%(asctime)-15s %(message)s",
            filename=os.path.join(project_path, "logs", "pr_cma_agent.log"),
        )

        # 初始化glove embedding
        # self.glove_embedding = InstructionEmbedding(
        #     source_folder=os.path.join(project_path, "data", "datasets", "real_deploy"),
        #     glove_path=os.path.join(project_path, "data", "glove", "glove.6B.50d.txt")
        # )
        self.glove_embedding = InstructionEmbedding(
            source_folder=os.path.join(project_path, "data", "datasets", "R2R_VLNCE_v1-3_preprocessed"),
            glove_path=os.path.join(project_path, "data", "glove", "glove.6B.50d.txt"),
        )

        # 初始化policy
        self.policy, _, _, _ = initialize_policy(
            self.eval_config,
            logger=self.logger,
            load_from_ckpt=True,
            device=self.device,
            load_from_pretrain=False,
            action_stats=None,
        )
        self.policy.eval()

        if self.eval_config.MODEL.policy_name == "CMA_CLIP_Policy":
            self.use_clip_encoders = True
        else:
            self.use_clip_encoders = False

        self.use_bert = False
        self.bert_tokenizer = None
        self.is_clip_long = False
        if self.use_clip_encoders:
            if self.eval_config.MODEL.TEXT_ENCODER.type == 'roberta':
                self.bert_tokenizer = BertTokenizer(
                    max_length=self.eval_config.MODEL.INSTRUCTION_ENCODER.max_length,
                    load_model=self.eval_config.MODEL.INSTRUCTION_ENCODER.load_model,
                    device=self.device,
                )
                self.use_bert = True
            elif self.eval_config.MODEL.TEXT_ENCODER.type == 'clip-long':
                self.bert_tokenizer = longclip.tokenize
                self.use_bert = True
                self.is_clip_long = True

        if hasattr(self.eval_config.MODEL, 'TEXT_ENCODER'):
            # use instr clip-long
            self.bert_tokenizer = longclip.tokenize
            self.use_bert = True
            self.is_clip_long = True
        self.reset_flag = True

    def eval_action(self, reset_flag, env_nums, rgb, depth, instruction, camera_pose=None, run_model=True):
        print("=====[EVAL_ACTION]=====")
        info = {}

        info['rgb'] = rgb
        info['depth'] = depth
        info['instruction'] = instruction

        print(f"instruction: {instruction}")

        if camera_pose is not None:
            camera_position = camera_pose[:3, 3]
            camera_rotation = camera_pose[:3, :3]
            # change from matrix to euler angle(in rad)
            from scipy.spatial.transform import Rotation as R

            # change from matrix to 四元数
            camera_rotation_quat = R.from_matrix(camera_rotation).as_quat()
            camera_rotation_quat = camera_rotation_quat.tolist()
            camera_rotation_quat = [
                camera_rotation_quat[3],
                camera_rotation_quat[0],
                camera_rotation_quat[1],
                camera_rotation_quat[2],
            ]
            camera_yaw = R.from_quat(camera_rotation_quat).as_euler('xyz', degrees=False)[2]
            info['camera_position'] = camera_position
            info['camera_rotation'] = camera_rotation_quat
            info['camera_yaw'] = camera_yaw

        return self.step(info, reset_flag, test_verbose=True)

    def reset_nn_states(self, env_nums=1):
        self.rnn_states = torch.zeros(
            env_nums,
            self.policy.num_recurrent_layers,
            self.eval_config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(env_nums, 1, device=self.device, dtype=torch.long)
        self.not_done_masks = torch.zeros(env_nums, 1, dtype=torch.uint8, device=self.device)
        self.stop = False

    def reset(self):
        self.reset_nn_states(env_nums=1)

    def step(self, info, reset_flag, test_verbose=False):
        """
        根据目标点、图像和深度预测下一个离散动作
        返回值: 0(stop), 1(forward), 2(left), 3(right)
        """
        self.policy.eval()
        env_nums = 1
        if reset_flag:
            print("=====[RESET]=====")
            self.reset_nn_states(env_nums=1)
            # self.reset_flag = False

        rgb = info['rgb']
        depth = info['depth']
        instruction = info['instruction']  # TODO: server or client to provide instruction?

        # observations = get_obs(instruction=instruction, rgb=rgb, depth=depth)
        observations = get_obs(instruction=instruction, rgb=rgb, depth=depth)
        observations = extract_instruction_tokens(
            observations=observations,
            bert_tokenizer=self.bert_tokenizer,
            is_clip_long=self.is_clip_long,
            glove_embedding=self.glove_embedding,
        )

        observations = batch_obs(observations, self.device)
        observations["steps"] = torch.zeros(1, device=self.device)
        # self.reset()
        # 构建policy输入
        batch = {
            'mode': 'inference',
            'observations': observations,
            'rnn_states': self.rnn_states,
            'prev_actions': self.prev_actions,
            'masks': self.not_done_masks,
            'pe_finetuned': self.pe_finetuned,
        }

        if test_verbose:
            import matplotlib.pyplot as plt

            plt.imsave('logs/pr-test.jpg', observations[0]['rgb'].cpu().numpy())

        if self.use_clip_encoders:
            batch.update(
                {
                    "need_img_extraction": True,
                    "img_mod": self.eval_config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                    'proj': self.eval_config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                    'process_images': True,
                    'need_txt_extraction': True,
                    "depth_return_x_before_fc": False,
                }
            )

        with torch.no_grad():
            actions, self.rnn_states, _ = self.policy(batch)

        dones = [False]
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.tensor(
            [[0] if done else [1] for done in dones],
            dtype=torch.uint8,
            device=self.device,
        )

        if actions[0] == 0:
            self.stop = True

        if self.stop:
            print("=====[STOP]=====")
            actions = torch.tensor([0], device=self.device)

        return actions.cpu().numpy()[0]

    def _get_config(self, project_path, model_type, pe_finetuned):
        """返回默认配置"""
        # cfg_file_path = f"{project_path}/vln/configs/v2/pr_eval.json"
        # cfg_file_path = f"{project_path}/vln/configs/v2/eval_sixth_floor_cma_clip.json"
        if model_type == 'cma_clip':
            cfg_file_path = f"{project_path}/vln/configs/v2/eval_cma.json"
        elif model_type == 'seq2seq':
            cfg_file_path = f"{project_path}/vln/configs/v2/eval_seq2seq.json"
        with open(cfg_file_path, 'r') as file:
            config = json.load(file)
        self.config = config
        name = config["name"]
        robot_name = config["robot_name"]  # h1 / aliengo
        eval_config_path = config["eval_cfg_file"]

        self.eval_config = get_eval_config(project_path=None, ckpt_to_load=None, eval_config_path=eval_config_path)

        if not pe_finetuned:
            self.eval_config.IL.ckpt_to_load = self.eval_config.IL.original_ckpt_to_load
            print(f"load original ckpt: {self.eval_config.IL.original_ckpt_to_load}")
        else:
            print(f"load finetuned ckpt: {self.eval_config.IL.ckpt_to_load}")


if __name__ == "__main__":
    agent = PR_CMA_Agent(project_path="/home/pjlab/w61/w61_grutopia")

    rgb = np.load("data/real_deploy/data/rgb/0.npy")
    depth = np.load("data/real_deploy/data/depth/0.npy")
    instruction = "turn left and walk to the door"
    camera_pose = None

    action = agent.eval_action(env_nums=1, rgb=rgb, depth=depth, instruction=instruction, camera_pose=camera_pose)
    print(action)
