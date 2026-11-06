"""The training workers, for methods that read with the actor's own model.

Rollout, judging and the update run one after the other, so while vLLM
generates, a method may borrow a training worker: its process holds the
actor's peft model on the GPU. `on_a_worker` runs a function there from a
rollout (`fn(model, *args)`), `on_every_worker` runs one on all of them at
once from the trainer, as collectives need.
"""

import random
import threading
from contextlib import contextmanager

import ray

DESK = "oi-workers"
"""The Ray actor that tells rollouts where the training workers are."""


@ray.remote
class Desk:
    """Holds the training workers' handles."""

    def __init__(self, workers: list):
        """Keep the handles."""
        self.workers = workers

    def get(self) -> list:
        """The handles."""
        return self.workers


def register(workers: list):
    """Make the training workers known to the rollouts; the trainer keeps
    the returned handle alive.
    """
    return Desk.options(name=DESK, get_if_exists=True).remote(list(workers))


WORKERS: list = []
"""This process's handles on the training workers."""
LOCK = threading.Lock()
"""In a worker's process: one method call on the model at a time."""


def model(instance):
    """In a training worker's process: the actor's peft model."""
    workers = getattr(instance, "worker_dict", None) or {"": instance}
    worker = next(w for w in workers.values() if hasattr(w, "actor"))
    engine = worker.actor.engine
    if getattr(engine, "_is_offload_param", False):
        raise RuntimeError("reading with the actor needs it on the GPU")
    return engine.module


@contextmanager
def last_logits(model, keep: int):
    """The model's logits for its last `keep` positions only: its output
    head reads no others. verl's patched forward ignores
    `logits_to_keep`, and a long sequence's logits fill a GPU.
    """
    head = model.get_output_embeddings()
    hook = head.register_forward_pre_hook(
        lambda module, args: (args[0][..., -keep:, :],)
    )
    try:
        yield
    finally:
        hook.remove()


def run(instance, fn, *args):
    """`fn(model, *args)` with the worker's actor model, alone on it."""
    with LOCK:
        return fn(model(instance), *args)


async def on_a_worker(fn, *args):
    """`fn(model, *args)` on a random training worker, from a rollout."""
    if not WORKERS:
        WORKERS.extend(await ray.get_actor(DESK).get.remote())
    worker = random.choice(WORKERS)
    return await worker.__ray_call__.remote(run, fn, *args)


def on_every_worker(workers: list, fn, *args) -> list:
    """`fn(model, *args)` on every training worker at once."""
    return ray.get([w.__ray_call__.remote(run, fn, *args) for w in workers])
