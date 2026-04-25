if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import json

import click
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('--input', '-i', required=True)
@click.option('--dt', default=0.1, type=float)
def main(input, dt):
    buffer = ReplayBuffer.create_from_path(input)
    lengths = buffer.episode_lengths
    durations = lengths * dt
    result = {'duration/mean': np.mean(durations)}

    text = json.dumps(result, indent=2)
    print(text)


if __name__ == '__main__':
    main()
