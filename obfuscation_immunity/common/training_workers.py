"""
Calls into verl's training workers, for judges that use the actor's model.

verl's training workers are Ray actors whose processes hold the actor's
peft model on the GPU. verl's synchronous trainer runs one phase at a time,
so while vLLM generates they sit idle, and a judge may run a function
on one of them from a rollout (`call_on_a_training_worker`), or on all of
them at once from the trainer, as collectives need
(`call_on_every_training_worker`). The function gets the actor's model as
its first argument.
"""

import random
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import ray
import torch
from peft import PeftModel
from ray.actor import ActorHandle

TRAINING_WORKER_DIRECTORY_NAME = "oi-training-workers"
"""The name of the Ray actor through which rollouts find the workers."""

TRAINING_WORKER_HANDLES: list[ActorHandle] = []
"""This process's handles on the training workers, fetched on first use."""

ACTOR_MODEL_LOCK = threading.Lock()
"""In a training worker's process: one of our calls on the model at once."""


@ray.remote
class TrainingWorkerDirectory:
    """
    Holds the training workers' Ray handles, for rollouts to fetch.

    Attributes:
        worker_handles: The handles.
    """

    worker_handles: list[ActorHandle]

    def __init__(self, worker_handles: list[ActorHandle]):
        """
        Keeps the handles.

        Args:
            worker_handles: verl's training workers.
        """
        self.worker_handles = worker_handles

    def get(self) -> list[ActorHandle]:
        """
        Gives the handles.

        Returns:
            verl's training workers.
        """
        return self.worker_handles


def publish_training_workers(worker_handles: list[ActorHandle]) -> ActorHandle:
    """
    Publishes the training workers under a name rollouts can look up.

    Args:
        worker_handles: verl's training workers.

    Returns:
        The directory's handle; the caller keeps it, so that it stays alive.
    """
    directory = TrainingWorkerDirectory.options(
        name=TRAINING_WORKER_DIRECTORY_NAME, get_if_exists=True
    )
    return directory.remote(list(worker_handles))


def actor_model_of(worker: Any) -> PeftModel:
    """
    Finds the actor's peft model inside a training worker.

    Args:
        worker: The training worker's Ray actor, in its own process.

    Returns:
        The actor's peft model.

    Raises:
        RuntimeError: verl offloaded the model from the GPU.
    """
    roles = getattr(worker, "worker_dict", None) or {"": worker}
    actor_worker = next(w for w in roles.values() if hasattr(w, "actor"))
    engine = actor_worker.actor.engine
    if getattr(engine, "_is_offload_param", False):
        raise RuntimeError("using the actor's model needs it on the GPU")
    return engine.module


@contextmanager
def logits_for_last_positions(
    model: torch.nn.Module, positions: int
) -> Generator[None]:
    """
    Makes the model's output head compute logits for its last positions only.

    verl's patched forward ignores `logits_to_keep`, and a long sequence's
    logits over the whole vocabulary fill a GPU.

    Args:
        model: The model.
        positions: How many of the last positions to keep.

    Yields:
        Nothing; inside, the model's logits cover the last `positions`.
    """
    head = model.get_output_embeddings()
    hook = head.register_forward_pre_hook(
        lambda module, inputs: (inputs[0][..., -positions:, :],)
    )
    try:
        yield
    finally:
        hook.remove()


def call_with_actor_model(
    worker: Any, function: Callable[..., Any], *args: Any
) -> Any:
    """
    Calls a function with the worker's actor model, one call at a time.

    Args:
        worker: The training worker's Ray actor, in its own process.
        function: Called as `function(model, *args)`.
        *args: The function's other arguments.

    Returns:
        What the function returns.
    """
    with ACTOR_MODEL_LOCK:
        return function(actor_model_of(worker), *args)


async def call_on_a_training_worker(
    function: Callable[..., Any], *args: Any
) -> Any:
    """
    Calls a function with the actor's model on a random training worker.

    Args:
        function: Called as `function(model, *args)` on the worker.
        *args: The function's other arguments.

    Returns:
        What the function returns.
    """
    if not TRAINING_WORKER_HANDLES:
        directory = ray.get_actor(TRAINING_WORKER_DIRECTORY_NAME)
        TRAINING_WORKER_HANDLES.extend(await directory.get.remote())
    worker = random.choice(TRAINING_WORKER_HANDLES)
    return await worker.__ray_call__.remote(
        call_with_actor_model, function, *args
    )


def call_on_every_training_worker(
    worker_handles: list[ActorHandle], function: Callable[..., Any], *args: Any
) -> list[Any]:
    """
    Calls a function with the actor's model on every training worker at once.

    Args:
        worker_handles: verl's training workers.
        function: Called as `function(model, *args)` on each worker.
        *args: The function's other arguments.

    Returns:
        What the function returned on each worker.
    """
    calls = [
        worker.__ray_call__.remote(call_with_actor_model, function, *args)
        for worker in worker_handles
    ]
    return ray.get(calls)
