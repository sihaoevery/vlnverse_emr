from typing import List, Optional

from pydantic import BaseModel


class EvalCfg(BaseModel, extra='allow'):
    use_ckpt_config: Optional[bool] = None
    save_results: Optional[bool] = None
    split: Optional[List[str]] = None
    ckpt_to_load: Optional[str] = None
    max_steps: Optional[int] = None
    action: Optional[str] = None
    sample: Optional[bool] = None
    success_distance: Optional[float] = None
    rotation_threshold: Optional[float] = None
    num_sample: Optional[int] = None
    start_eval_epoch: Optional[int] = None
    stop_mode: Optional[str] = None
    pm_threshold: Optional[float] = None
    step_interval: Optional[int]
    len_traj_act: Optional[int] = None
