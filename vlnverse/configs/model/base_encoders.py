from typing import List, Optional

from pydantic import BaseModel


class TextEncoder(BaseModel, extra='allow'):
    load_model: Optional[bool]
    max_length: Optional[int]
    update_text_encoder: Optional[bool]
    type: Optional[str]
    model_name: Optional[str]
    model_path: Optional[str]
    num_l_layers: Optional[int]
    hidden_size: Optional[int]
    vocab_size: Optional[int]
    embedding_size: Optional[int]
    sot_token: Optional[int]
    eot_token: Optional[int]
    pad_token: Optional[int]
    ablate: Optional[bool] = None
    final_state_only: Optional[bool] = None


class ImageEncoderRgb(BaseModel, extra='allow'):
    load_model: Optional[bool]
    update_rgb_encoder: Optional[bool]
    model_name: Optional[str]
    model_path: Optional[str]
    rgb_proj: Optional[bool]
    feature_dim: Optional[int]
    projection_dim: Optional[int]
    img_mod: Optional[str]
    multi_patches_num: Optional[int]


class ImageEncoderDepth(BaseModel, extra='allow'):
    load_model: Optional[bool]
    update_depth_encoder: Optional[bool]
    bottleneck: Optional[str]
    feature_dim: Optional[int]
    projection_dim: Optional[int]
    cnn_type: Optional[str]
    output_size: Optional[int]
    ddppo_checkpoint: Optional[str]
    backbone: Optional[str]


class ImageEncoder(BaseModel, extra='allow'):
    use_stack: Optional[bool]
    dropout: Optional[float]
    use_env_drop: Optional[bool]
    env_drop: Optional[float]
    rgb: Optional[ImageEncoderRgb] = None
    depth: Optional[ImageEncoderDepth] = None


class CrossModalEncoder(BaseModel, extra='allow'):
    load_model: Optional[bool]
    input_type: Optional[int]
    num_x_layers: Optional[int]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    txt_to_img: Optional[bool]
    txt_to_img_layer: Optional[int]


class StateEncoder(BaseModel, extra='allow'):
    hidden_size: Optional[int]
    rnn_type: Optional[str]
    num_recurrent_layers: Optional[int] = None
    rgb_depth_embed_method: Optional[str] = None
    use_dropout: Optional[bool] = None
    dropout_rate: Optional[float] = None


class ProgressMonitor(BaseModel, extra='allow'):
    use: Optional[bool]
    concat_state_txt: Optional[bool] = None
    alpha: Optional[float] = None


class DiffusionPolicyDistance(BaseModel, extra='allow'):
    min_dist_cat: Optional[int]
    max_dist_cat: Optional[int]


class DiffusionPolicyAction(BaseModel, extra='allow'):
    min_dist_cat: Optional[int]
    max_dist_cat: Optional[int]


class DiffusionPolicyActionStats(BaseModel, extra='allow'):
    min: Optional[List[float]]
    max: Optional[List[float]]


class DiffusionPolicy(BaseModel, extra='allow'):
    use: Optional[bool]
    type: Optional[str]
    scheduler: Optional[str]
    pred_type: Optional[str]
    clip_sample: Optional[bool]
    use_cls_free_guidance: Optional[bool]
    cls_free_guidance_scale: Optional[float]
    cls_mask_ratio: Optional[float]
    random_mask_rgb: Optional[bool]
    random_mask_instr: Optional[bool]
    cls_mask_method: Optional[str]
    action_stats: Optional[DiffusionPolicyActionStats]
    metric_waypoint_spacing: Optional[int]
    num_diffusion_iters: Optional[int]
    transformer_n_cond_layers: Optional[int]
    transformer_n_layers: Optional[int]
    transformer_encoding_size: Optional[int]
    transformer_p_drop_emb: Optional[float]
    txt_len: Optional[int]
    waypoint_spacing: Optional[int]
    len_traj_pred: Optional[int]


class DistancePredictor(BaseModel, extra='allow'):
    use: Optional[bool]
    normalize: Optional[bool]


class ImuEncoder(BaseModel, extra='allow'):
    input_size: Optional[int]
    encoding_size: Optional[int]
    use: Optional[bool]
    to_local_coords: Optional[bool]


class PrevActionEncoder(BaseModel, extra='allow'):
    input_size: Optional[int] = None
    encoding_size: Optional[int]
    use: Optional[bool] = None
    to_local_coords: Optional[bool] = None


class StopProgressPredictor(BaseModel, extra='allow'):
    use: Optional[bool]
    concat_state_txt: Optional[bool]
    type: Optional[str]
    loss_alpha: Optional[int]


class InstructionEncoder(BaseModel, extra='allow'):
    sensor_uuid: Optional[str]
    vocab_size: Optional[int]
    use_pretrained_embeddings: Optional[bool]
    embedding_file: Optional[str]
    dataset_vocab: Optional[str]
    fine_tune_embeddings: Optional[bool]
    embedding_size: Optional[int]
    hidden_size: Optional[int]
    rnn_type: Optional[str]
    final_state_only: Optional[bool] = None
    bidirectional: Optional[bool]
    max_length: Optional[int] = None
    load_model: Optional[bool] = None


class RgbEncoder(BaseModel, extra='allow'):
    cnn_type: Optional[str]
    output_size: Optional[int]
    trainable: Optional[bool]


class DepthEncoder(BaseModel, extra='allow'):
    cnn_type: Optional[str]
    output_size: Optional[int]
    backbone: Optional[str]
    ddppo_checkpoint: Optional[str]
    trainable: Optional[bool]


class Seq2Seq(BaseModel, extra='allow'):
    use_prev_action: Optional[bool]


class ModelCfg(BaseModel, extra='allow'):
    policy_name: Optional[str]
    ablate_instruction: Optional[bool] = None
    ablate_depth: Optional[bool] = None
    ablate_rgb: Optional[bool] = None
    normalize_rgb: Optional[bool] = None
    max_step: Optional[int] = None
    learn_angle: Optional[bool] = None
    len_traj_act: Optional[int] = None
    text_encoder: Optional[TextEncoder] = None
    image_encoder: Optional[ImageEncoder] = None
    cross_modal_encoder: Optional[CrossModalEncoder] = None
    state_encoder: Optional[StateEncoder]
    progress_monitor: Optional[ProgressMonitor] = None
    diffusion_policy: Optional[DiffusionPolicy] = None
    distance_predictor: Optional[DistancePredictor] = None
    imu_encoder: Optional[ImuEncoder] = None
    prev_action_encoder: Optional[PrevActionEncoder] = None
    stop_progress_predictor: Optional[StopProgressPredictor] = None

    instruction_encoder: Optional[InstructionEncoder] = None
    rgb_encoder: Optional[RgbEncoder] = None
    depth_encoder: Optional[DepthEncoder] = None
    seq2seq: Optional[Seq2Seq] = None
