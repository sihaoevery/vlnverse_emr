from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class RealPushTImageRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy):
        return dict()
