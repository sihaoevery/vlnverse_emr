from typing import List, Optional

from pydantic import BaseModel


class TaskCfg(BaseModel, extra='allow'):
    run_type: str = 'sample'
    sample_type: str = 'discrete'  # [continuous,discrete]
    scan: Optional[str] = None
    headless: bool = False
    split_data_types: List[str] = ['train']
    base_data_dir: str
    mp3d_data_dir: str
    name: str
    retry_list: List[str]
    total_rank: int
    max_round: int = 10
    filter_same_trajectory: bool = True
    flash: bool = False
    aperture: int = 200
    dagger_percentage: float = 0
    robot_height: float = 1.55
    fall_height_threshold: float = 0.5
    max_step: int = 25000
    robot_ankle_height: float = 0.0758
