from typing import List, Optional

from pydantic import BaseModel

from ..model.base_encoders import ModelCfg
from .eval import EvalCfg
from .il import IlCfg


class ExpCfg(BaseModel, extra='allow'):
    name: Optional[str] = None
    model_name: Optional[str] = None
    torch_gpu_id: Optional[int] = None
    torch_gpu_ids: Optional[List[int]] = None
    checkpoint_folder: Optional[str] = None
    log_dir: Optional[str] = None
    seed: Optional[int] = None
    eval: Optional[EvalCfg] = None
    il: Optional[IlCfg] = None
    model: Optional[ModelCfg] = None
