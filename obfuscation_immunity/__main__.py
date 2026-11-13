"""
Trains the policy against one judge: verl's PPO trainer on configs/.

    python -m obfuscation_immunity judge=actor_judge \
        model=qwen3.5-2b trainer.total_training_steps=2
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from obfuscation_immunity.common import puzzles
from obfuscation_immunity.common.trainer import PairTaskRunner


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    """
    Runs one training job from the composed config.

    Resolves the config while Hydra can (its `${hydra:…}` values), writes
    the puzzles if they are not there yet, has verl check the config, and
    trains.

    Args:
        config: configs/train.yaml with the chosen judge and model, and
            the command line's overrides.
    """
    OmegaConf.resolve(config)
    puzzle_file = Path(config.data.train_files)
    if not puzzle_file.exists():
        puzzles.write_puzzles(
            puzzle_file, config.puzzle.name, **config.puzzle.settings
        )
    auto_set_device(config)
    validate_config(
        config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=PairTaskRunner)


if __name__ == "__main__":
    main()
