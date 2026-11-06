"""verl's synchronous trainer and task runner, with a method in them: its
`train` after each actor update, and the run's own reporting
(common.progress): verl's logger gets no backend.
"""

import os

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync

from obfuscation_immunity.common import progress, workers
from obfuscation_immunity.common.base_method import BaseMethod, running


class Trainer(PPOTrainerSync):
    """verl's synchronous trainer, with the method's `train` after each
    actor update and the run's own logging.
    """

    def __init__(self, config, method: BaseMethod):
        """verl's trainer for `method`."""
        super().__init__(config)
        self.method = method

    def fit(self, agent_loop_manager):
        """verl's training loop, reported to the console and wandb."""
        self.progress = progress.Progress(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            self.method.kwargs,
            os.environ[progress.EPISODES],
        )
        try:
            super().fit(agent_loop_manager)
        finally:
            self.progress.finish()

    def on_init_end(self):
        """Make the training workers known to the rollouts; the method's
        `train` for the steps done so far (none, unless resumed).
        """
        super().on_init_end()
        self.desk = workers.register(self.actor_rollout_wg.workers)
        self.method.train(self, self.global_steps)

    def on_step_end(self):
        """verl's weight sync, then the method's `train`."""
        super().on_step_end()
        self.method.train(self, self.global_steps)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        """verl's numbers, then the step's report."""
        super()._compute_metrics(
            batch, metrics, timing_raw, global_steps, epoch
        )
        self.progress.step(global_steps, metrics)


@ray.remote(num_cpus=1)
class Runner:
    """verl's task runner, with the method's trainer."""

    def run(self, config) -> None:
        """Build the trainer and the rollouts, and train."""
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        try:
            running().prepare()
            trainer = Trainer(config, running())
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
        finally:
            tq.close()
