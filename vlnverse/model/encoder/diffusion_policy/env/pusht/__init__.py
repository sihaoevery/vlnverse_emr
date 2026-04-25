import diffusion_policy.env.pusht
from gym.envs.registration import register

register(
    id='pusht-keypoints-v0',
    entry_point='envs.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0,
)
