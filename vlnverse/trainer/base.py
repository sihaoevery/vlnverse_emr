# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import transformers
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)


class BaseTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if 'bias' not in name]
            optimizer_grouped_parameters = [
                {
                    'params': [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    'weight_decay': self.args.weight_decay,
                },
                {
                    'params': [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    'weight_decay': 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """
        save model to specified directory

        handle DDP wrapped model
        """
        # check if it is a DDP wrapped model
        if hasattr(self.model, 'module'):
            # get original model
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # save model
        model_to_save.save_pretrained(output_dir, state_dict=state_dict)

        # save tokenizer (if any)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # save trainer state
        # torch.save(self.state_dict(), os.path.join(output_dir, "trainer_state.pt"))
        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f'No valid checkpoint found in output directory ({self.args.output_dir})')

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        return super().train(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
        )  # 这里会调用transformer中的train()
