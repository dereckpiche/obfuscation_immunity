"""
verl's synchronous trainer and task runner, with the run's judge in them.

The judge may bring verl's actor worker its own class
(`Judge.actor_worker_class`), and acts once verl is ready
(`on_trainer_ready`) and after each actor update (`after_actor_update`). A
judge that learns sends its own trajectories to `JUDGE_PARTITION`
(`PairAgentLoopWorker`), where verl's own update trains it
(`PairTrainer.update_on_partition`). The trainer reports each step itself
(common.progress): verl's own logger is off.
"""

from typing import Any

import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from ray.actor import ActorHandle
from transfer_queue import KVBatchMeta
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.utils.debug import marked_timer
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from obfuscation_immunity.common import training_workers
from obfuscation_immunity.common.judge import (
    JUDGE_PARTITION,
    Judge,
    build_judge,
    whole_batches,
)
from obfuscation_immunity.common.progress import Progress
from obfuscation_immunity.common.transcript import PairRecord

AGENT_LOOP_WORKER_TQ = (
    AgentLoopWorkerTQ.__ray_metadata__.modified_class.__ray_actor_class__
)
"""verl's agent loop worker for the transfer queue, unwrapped from Ray."""

ACTOR_ROLES = (Role.ActorRollout, Role.ActorRolloutRef, Role.Actor)
"""The roles verl's actor worker may take."""


class PairAgentLoopWorker(AGENT_LOOP_WORKER_TQ):
    """verl's agent loop worker, also writing the judge's trajectories."""

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Builds verl's worker.

        Defined here because Ray ships an inherited `__init__` by value, with
        its own copy of verl's class, which its `super()` then rejects.

        Args:
            *args: verl's arguments for the worker.
            **kwargs: verl's keyword arguments for the worker.
        """
        super().__init__(*args, **kwargs)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput, validate: bool, **kwargs: Any
    ) -> None:
        """
        Puts the pair's trajectory in the queue, and the judge's beside it.

        Args:
            output: The pair's trajectory; its `extra_fields` may carry the
                judge's (`Verdict.judge_trajectory`).
            validate: Whether the trajectory is for validation.
            **kwargs: The dataset row and verl's sampling information.
        """
        judge_trajectory = output.extra_fields.pop("judge_trajectory", None)
        await super()._agent_loop_postprocess(output, validate, **kwargs)
        if judge_trajectory is not None and not validate:
            await self.put_judge_trajectory(judge_trajectory, output, kwargs)

    async def put_judge_trajectory(
        self,
        judge_trajectory: dict[str, Any],
        output: AgentLoopOutput,
        kwargs: dict[str, Any],
    ) -> None:
        """
        Puts the judge's trajectory in `JUDGE_PARTITION`, as verl puts its own.

        It takes the pair's key, so that GRPO groups it with the judge's
        other trajectories on the same puzzle; its score is its reward, and
        the judge's sampler log-probabilities are verl's rollout ones, for
        its correction of the rollout and training mismatch.

        Args:
            judge_trajectory: The judge's turn (`Verdict.judge_trajectory`).
            output: The pair's trajectory.
            kwargs: The dataset row and verl's sampling information.
        """
        judge_output = AgentLoopOutput(
            prompt_ids=judge_trajectory["prompt_ids"],
            response_ids=judge_trajectory["response_ids"],
            response_mask=judge_trajectory["response_mask"],
            response_logprobs=judge_trajectory["response_logprobs"],
            reward_score=judge_trajectory["score"],
            num_turns=2,
            metrics=output.metrics,
            extra_fields={
                key: output.extra_fields.get(key)
                for key in ("min_global_steps", "max_global_steps")
            },
        )
        prompts = torch.tensor(judge_output.prompt_ids, dtype=torch.int64)
        responses = torch.tensor(judge_output.response_ids, dtype=torch.int64)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        multi_modal_inputs = self._compute_multi_modal_inputs(
            judge_output, input_ids
        )
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            multi_modal_inputs,
        ).squeeze(0)
        field = judge_output.as_dict()
        field.update(kwargs)
        field.pop("multi_modal_data", None)
        field["loss_mask"] = field["response_mask"]
        field["input_ids"] = input_ids
        field["position_ids"] = position_ids
        field["multi_modal_inputs"] = multi_modal_inputs
        tag = {
            "status": "success",
            "prompt_len": len(prompts),
            "response_len": len(responses),
            "seq_len": len(input_ids),
            "global_steps": kwargs["global_steps"],
            "min_global_steps": judge_output.extra_fields["min_global_steps"],
            "max_global_steps": judge_output.extra_fields["max_global_steps"],
        }
        await tq.async_kv_batch_put(
            keys=[f"{kwargs['uid']}_{kwargs['session_id']}_0"],
            fields=list_of_dict_to_tensordict([field]),
            tags=[tag],
            partition_id=JUDGE_PARTITION,
        )


class PairAgentLoopManager(AgentLoopManagerTQ):
    """verl's agent loop manager, running `PairAgentLoopWorker`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Builds verl's manager, with `PairAgentLoopWorker` as its worker.

        Args:
            *args: verl's arguments for the manager.
            **kwargs: verl's keyword arguments for the manager.
        """
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(PairAgentLoopWorker)


class PairTrainer(PPOTrainerSync):
    """
    verl's synchronous trainer, with the run's judge and the report.

    Attributes:
        judge: The run's judge.
        progress: The run's console lines and wandb run.
        training_worker_directory: The Ray actor through which rollouts
            find the training workers; kept here so that it stays alive.
    """

    judge: Judge
    progress: Progress
    training_worker_directory: ActorHandle

    def __init__(self, config: DictConfig, judge: Judge, progress: Progress):
        """
        Builds verl's trainer, with the run's judge and report.

        Args:
            config: The run's whole config.
            judge: The run's judge.
            progress: The run's console lines and wandb run.
        """
        super().__init__(config)
        self.judge = judge
        self.progress = progress

    def _init_resource_pool_mgr(self) -> None:
        """Maps verl's roles to workers, the actor's to the judge's class."""
        super()._init_resource_pool_mgr()
        if self.judge.actor_worker_class is None:
            return
        for role in ACTOR_ROLES:
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(
                    self.judge.actor_worker_class
                )

    def on_init_end(self) -> None:
        """Syncs vLLM, publishes the training workers, readies the judge."""
        super().on_init_end()
        self.training_worker_directory = (
            training_workers.publish_training_workers(
                self.actor_rollout_wg.workers
            )
        )
        self.judge.on_trainer_ready(self)

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """
        Runs verl's step, the actor's update, then the judge's turn.

        Args:
            metrics: The step's numbers, to add to.
            timing_raw: verl's timings for the step.

        Returns:
            The step's trajectories, in verl's transfer queue.
        """
        batch = super().step(metrics, timing_raw)
        with marked_timer("judge_update", timing_raw):
            self.judge.after_actor_update(self, metrics)
        return batch

    def update_on_partition(self, partition_id: str) -> dict[str, float]:
        """
        Trains the actor's engine on a partition's trajectories, with verl.

        verl's own pipeline: balancing, old and reference log-probs,
        advantages with the rollout correction, then the update; then the
        trajectories are cleared. It trains on whole batches only, oldest
        first, and leaves the rest for the next step: verl pads any other
        size with synthetic samples whose multi-row position ids
        (Qwen3.5's) break its micro-batching.

        Args:
            partition_id: The transfer queue partition to train on.

        Returns:
            verl's numbers for the update; none if there is not yet a
            whole batch.
        """
        listed = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        metrics: dict[str, float] = {}
        keys = whole_batches(listed, self.batch_multiple())
        if not keys:
            return metrics
        batch = KVBatchMeta(
            partition_id=partition_id,
            keys=keys,
            tags=[listed[key] for key in keys],
        )
        temperature = self.config.actor_rollout_ref.rollout.temperature
        batch.extra_info["temperature"] = temperature
        batch = self._balance_batch(batch, metrics=metrics)
        batch = self._compute_old_log_prob(batch, metrics=metrics)
        if self.use_reference_policy:
            batch = self._compute_ref_log_prob(batch, metrics=metrics)
        batch = self._compute_advantage(batch, metrics=metrics)
        batch = self._update_actor(batch, metrics=metrics)
        tq.kv_clear(keys=batch.keys, partition_id=partition_id)
        return metrics

    def batch_multiple(self) -> int:
        """
        Gives the batch size verl's update needs, as it computes it.

        Returns:
            The multiple of trajectories verl trains on without padding:
            the actor's mini-batch, and a share for each data-parallel rank.
        """
        workers = self.actor_rollout_wg
        if "actor" not in workers._dispatch_info:
            workers._dispatch_info["actor"] = workers._query_dispatch_info(
                "actor"
            )
        dp_size = max(workers._dispatch_info["actor"]) + 1
        return self._get_required_batch_multiple(dp_size)

    def _compute_metrics(
        self,
        batch: KVBatchMeta,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
        global_steps: int,
        epoch: int,
    ) -> None:
        """
        Computes verl's numbers for the step, then reports the step.

        Args:
            batch: The step's trajectories, in verl's transfer queue.
            metrics: The step's numbers so far.
            timing_raw: verl's timings for the step.
            global_steps: The step's number.
            epoch: The epoch's number.
        """
        super()._compute_metrics(
            batch, metrics, timing_raw, global_steps, epoch
        )
        stored = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        records = [
            PairRecord.from_dict(fields["pair_record"])
            for fields, tag in zip(stored["extra_fields"], batch.tags)
            if not tag.get("is_padding", False)
        ]
        self.progress.log_step(global_steps, metrics, records, self.tokenizer)


@ray.remote(num_cpus=1)
class PairTaskRunner:
    """verl's task runner, building the pair game's trainer."""

    def run(self, config: DictConfig) -> None:
        """
        Builds the judge, the report, the trainer and the rollouts.

        Then trains, and closes the report whatever happens.

        Args:
            config: The run's whole config, resolved.
        """
        tq.init(config.transfer_queue)
        progress = Progress(config)
        try:
            judge = build_judge(config.judge)
            judge.before_verl_starts(config)
            trainer = PairTrainer(config, judge, progress)
            trainer.init()
            manager = PairAgentLoopManager.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
        finally:
            progress.finish()
            tq.close()
