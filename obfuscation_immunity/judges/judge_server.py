"""
verl's vLLM server, able to serve a judge's LoRA named per request.

An adapter judge generates with its own LoRA (judges.adapter_judge
`write_served_adapter`), so that it thinks at vLLM's speed. A request whose
sampling parameters carry a `lora_dir` is served with the LoRA in that
directory; every other request with verl's own, the actor's.
"""

import argparse
import contextvars
from typing import Any

import ray
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer,
    vLLMReplica,
)
from vllm.lora.request import LoRARequest

REQUESTED_LORA_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "requested_lora_dir", default=None
)
"""The LoRA directory the request in progress asked for, if any."""

FIRST_EXTRA_LORA_ID = 1000
"""vLLM's id for the first extra LoRA; verl's own LoRA has id 123."""


class TwoLoraVLLMServer(vLLMHttpServer):
    """
    verl's vLLM server, with room for a second LoRA chosen per request.

    Attributes:
        lora_id_by_dir: vLLM's id for each LoRA directory served so far.
    """

    lora_id_by_dir: dict[str, int]

    async def run_server(self, args: argparse.Namespace) -> None:
        """
        Starts vLLM with the config's LoRA settings and per-request LoRAs.

        verl sets vLLM's LoRA settings for the actor's adapter alone; this
        puts back the config's (`engine_kwargs.vllm`: `max_loras`,
        `max_cpu_loras`, `max_lora_rank`), then wraps vLLM's generate so
        that a request's `lora_dir` picks its LoRA.

        Args:
            args: vLLM's engine arguments, as verl built them.
        """
        lora_settings = self.config.get("engine_kwargs", {}).get("vllm", {})
        for key in ("max_loras", "max_cpu_loras", "max_lora_rank"):
            setattr(args, key, lora_settings[key])
        await super().run_server(args)
        self.lora_id_by_dir = {}
        vllm_generate = self.engine.generate

        def generate_with_requested_lora(
            *args: Any, lora_request: LoRARequest | None = None, **kwargs: Any
        ) -> Any:
            """
            Runs vLLM's generate, with the requested LoRA if there is one.

            Args:
                *args: vLLM's arguments.
                lora_request: verl's LoRA, the actor's.
                **kwargs: vLLM's keyword arguments.

            Returns:
                What vLLM's generate returns.
            """
            lora_dir = REQUESTED_LORA_DIR.get()
            if lora_dir:
                next_id = FIRST_EXTRA_LORA_ID + len(self.lora_id_by_dir)
                lora_id = self.lora_id_by_dir.setdefault(lora_dir, next_id)
                lora_request = LoRARequest(lora_dir, lora_id, lora_dir)
            return vllm_generate(*args, lora_request=lora_request, **kwargs)

        self.engine.generate = generate_with_requested_lora

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        **kwargs: Any,
    ) -> Any:
        """
        Runs verl's generate, with the LoRA the request names.

        Args:
            prompt_ids: The prompt's tokens.
            sampling_params: The request's sampling parameters; its
                `lora_dir`, if any, is taken out.
            request_id: The request's id.
            **kwargs: verl's other arguments.

        Returns:
            What verl's generate returns.
        """
        lora_dir = sampling_params.pop("lora_dir", None)
        token = REQUESTED_LORA_DIR.set(lora_dir)
        try:
            return await super().generate(
                prompt_ids, sampling_params, request_id, **kwargs
            )
        finally:
            REQUESTED_LORA_DIR.reset(token)


class TwoLoraVLLMReplica(vLLMReplica):
    """verl's vLLM replica, running TwoLoraVLLMServer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Builds verl's replica, with TwoLoraVLLMServer as its server.

        Args:
            *args: verl's arguments for a replica.
            **kwargs: verl's keyword arguments for a replica.
        """
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TwoLoraVLLMServer)


def use_two_lora_vllm_servers() -> None:
    """Registers TwoLoraVLLMReplica as verl's vLLM replica, before it starts."""
    RolloutReplicaRegistry.register("vllm", lambda: TwoLoraVLLMReplica)
