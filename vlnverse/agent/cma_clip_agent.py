import time

import numpy as np
import torch
from gym import spaces

from vlnverse.agent.base import Agent
from vlnverse.configs.agent import AgentCfg
from vlnverse.configs.model.base_encoders import ModelCfg
from vlnverse.evaluator.utils.common import set_seed_model
from vlnverse.evaluator.utils.models import batch_obs
from vlnverse.model import get_config, get_policy


@Agent.register('cma_clip')
class CmaCLIPAgent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, agent_config: AgentCfg):
        super().__init__(agent_config)
        self._model_settings = ModelCfg(**agent_config.model_settings)
        self.tokenizer = None
        self.max_instr_len = 200
        if self._model_settings.text_encoder is not None and self._model_settings.text_encoder.type == 'clip-long':
            from vlnverse.model.basemodel.LongCLIP.model import longclip
            self.tokenizer = longclip.tokenize
            self.max_instr_len = getattr(self._model_settings.text_encoder, 'max_length', 248) or 248
            print('[CmaCLIPAgent] Using CLIP-Long on-the-fly tokenizer')
        else:
            print('[CmaCLIPAgent] Using pre-tokenized instruction_tokens from data')
        model_settings = self._model_settings
        set_seed_model(0)
        env_num = getattr(self._model_settings, 'env_num', 1)
        proc_num = getattr(self._model_settings, 'proc_num', 1)
        self.device = torch.device('cuda', 0)
        policy = get_policy(model_settings.policy_name)
        self.policy = policy.from_pretrained(
            agent_config.ckpt_path,
            config=get_config(self._model_settings.policy_name)(model_cfg={'model': self._model_settings.model_dump()}),
        ).to(self.device)

        # step required
        self._env_nums = env_num
        self._proc_num = proc_num
        self._rnn_states = torch.zeros(
            self._env_nums * self._proc_num,
            self.policy.num_recurrent_layers,
            self._model_settings.state_encoder.hidden_size,
            device=self.device,
        )
        self._prev_actions = torch.zeros(
            self._env_nums * self._proc_num,
            1,
            device=self.device,
            dtype=torch.long,
        )
        self._not_done_masks = torch.tensor([0 * self._env_nums * self._proc_num], device=self.device).to(torch.bool)

    def reset(self, reset_ls=None):
        if reset_ls is not None and len(reset_ls) > 0:
            print(f'CmaPolicyAgent{reset_ls} reset')

        if reset_ls is None:
            self._rnn_states = torch.zeros(
                self._env_nums * self._proc_num,
                self.policy.num_recurrent_layers,
                self._model_settings.state_encoder.hidden_size,
                device=self.device,
            )
            self._prev_actions = torch.zeros(
                self._env_nums * self._proc_num,
                1,
                device=self.device,
                dtype=torch.long,
            )
            self._not_done_masks = torch.zeros(
                self._env_nums * self._proc_num,
                1,
                device=self.device,
                dtype=torch.bool,
            )

        elif len(reset_ls) > 0:
            self._rnn_states.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._prev_actions.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._not_done_masks.index_fill_(
                dim=0,
                index=torch.tensor(reset_ls).to(self.device),
                value=False,
            )

    def inference(self, obs):
        start = time.time()

        # process change to here
        for ob in obs:
            instr_text = ob.get('instruction', None)
            instr_tokens = ob.get('instruction_tokens', None)

            if self.tokenizer is not None and isinstance(instr_text, str):
                # CLIP on-the-fly tokenization from raw text — always preferred when available
                tokens = self.tokenizer(instr_text)[0].tolist()
                ob['instruction'] = tokens
            else:
                # Fallback to pre-tokenized (GloVe or legacy)
                tokens = instr_tokens if instr_tokens is not None else instr_text
                if isinstance(tokens, dict):
                    raise ValueError(
                        f"instruction_tokens is a dict (keys={list(tokens.keys())}), "
                        "but no tokenizer is configured. Ensure instruction_type resolves "
                        "tokens before reaching the agent, or configure a CLIP tokenizer."
                    )
                if tokens is None or not isinstance(tokens, (list, torch.Tensor, np.ndarray)) or len(tokens) == 0:
                    raise ValueError(
                        f"Invalid instruction tokens: got {type(tokens)}. "
                        "Provide pre-tokenized data or configure a CLIP tokenizer via model_settings."
                    )
                instr = torch.as_tensor(tokens).long()
                instr = instr[:self.max_instr_len]
                ob['instruction'] = torch.nn.functional.pad(instr, (0, self.max_instr_len - instr.shape[0]), 'constant', 0)
            ob.pop('instruction_tokens', None)
        obs = batch_obs(obs, device=self.device)

        # need to change
        batch = {
            'mode': 'inference',
            'observations': obs,
            'rnn_states': self._rnn_states,
            'prev_actions': self._prev_actions,
            'masks': self._not_done_masks,
        }

        with torch.no_grad():
            actions, self._rnn_states, _ = self.policy.forward(batch)
        # 确保 actions 的形状与 _prev_actions 匹配
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # 从 [2] 变为 [1, 2]
            print(f'actions: {actions}\n prev_actions: {self._prev_actions}')
        self._prev_actions.copy_(actions)
        self._not_done_masks = torch.ones(
            self._env_nums * self._proc_num,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        end = time.time()
        print(f'CmaAgent step time: {round(end-start,4)}s')
        return actions.cpu().numpy().tolist()

    def step(self, obs):
        print('CmaPolicyAgent step')
        start = time.time()
        action = self.inference(obs)
        end = time.time()
        print(f'Time: {round(end-start,4)}s')
        return action
