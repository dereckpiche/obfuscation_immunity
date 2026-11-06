"""verl's vLLM server with a second LoRA adapter, the judge's: π and ψ
stacked into one (coevolving_judge.write), so that J_{ψ,π} thinks at
vLLM's speed. A request whose sampling params name an adapter directory
(`adapter`) is served with that adapter; every other request with verl's,
the actor's π.
"""

import contextvars

import ray
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer,
    vLLMReplica,
)
from vllm.lora.request import LoRARequest

ADAPTER = contextvars.ContextVar("adapter", default=None)
"""The adapter directory the request in progress asked for, if any."""


class JudgeServer(vLLMHttpServer):
    """verl's vLLM server, with room for the judge's adapter next to π."""

    async def run_server(self, args):
        """Start vLLM with two adapters' room, of the stacked rank, then
        let each request's `adapter` choose its LoRA.
        """
        from obfuscation_immunity.common.base_method import running

        method = running()
        args.max_loras = 2
        args.max_cpu_loras = 4
        rank = method.lora_rank + method.psi_rank
        args.max_lora_rank = get_vllm_max_lora_rank(rank)
        await super().run_server(args)
        self.adapters = {}
        generate = self.engine.generate

        def judged(*args, lora_request=None, **kwargs):
            """vLLM's generate, with the judge's adapter if asked for."""
            if path := ADAPTER.get():
                number = self.adapters.setdefault(
                    path, 1000 + len(self.adapters)
                )
                lora_request = LoRARequest(path, number, path)
            return generate(*args, lora_request=lora_request, **kwargs)

        self.engine.generate = judged

    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        """verl's generate, with the adapter the request names."""
        token = ADAPTER.set(sampling_params.pop("adapter", None))
        try:
            return await super().generate(
                prompt_ids, sampling_params, request_id, **kwargs
            )
        finally:
            ADAPTER.reset(token)


class JudgeReplica(vLLMReplica):
    """verl's vLLM replica, on JudgeServer."""

    def __init__(self, *args, **kwargs):
        """verl's replica, its servers the judge's."""
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(JudgeServer)


def register() -> None:
    """Have verl start its vLLM replicas on JudgeServer."""
    RolloutReplicaRegistry.register("vllm", lambda: JudgeReplica)
