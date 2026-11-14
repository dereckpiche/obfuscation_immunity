"""
A vLLM plugin: keeps a model's grouped RMSNorms in the Transformers backend.

vLLM runs a model it has no implementation of (K2-Horizon in vLLM 0.24)
through its Transformers backend, which swaps every `*RMSNorm` module for
its own. Its own normalizes over the whole hidden state; K2-Horizon's
normalizes in groups (`n_groups`). The swapped model is another model: vLLM
samples from it while verl trains the real one, and their log-probabilities
part (`mismatch` 0.14, where Qwen's is 0.0004). vLLM loads this in each of
its processes (pyproject.toml's `vllm.general_plugins` entry point).
"""

import logging
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)
"""Where the plugin says that it kept a grouped RMSNorm."""

BACKEND_MODULES = (
    "vllm.model_executor.models.transformers.utils",
    "vllm.model_executor.models.transformers.base",
)
"""The modules that hold vLLM's RMSNorm swap: its definition, and its use."""


def keep_grouped_rms_norms() -> None:
    """Makes vLLM's Transformers backend keep RMSNorms that work in groups."""
    from vllm.model_executor.models.transformers import utils

    swap_rms_norm = utils.replace_rms_norm_class
    if getattr(swap_rms_norm, "keeps_grouped_rms_norms", False):
        return

    def swap_ungrouped_rms_norm(rms_norm: Any, hidden_size: int) -> Any:
        """
        Swaps an RMSNorm for vLLM's, unless it normalizes in groups.

        Args:
            rms_norm: The model's RMSNorm module.
            hidden_size: The model's hidden size.

        Returns:
            The module itself if it has more than one group; else vLLM's.
        """
        if getattr(rms_norm, "n_groups", 1) > 1:
            if not swap_ungrouped_rms_norm.kept:
                LOGGER.warning(
                    "keeping the model's own %s (%d groups) in vLLM",
                    type(rms_norm).__name__,
                    rms_norm.n_groups,
                )
            swap_ungrouped_rms_norm.kept += 1
            return rms_norm
        return swap_rms_norm(rms_norm, hidden_size)

    swap_ungrouped_rms_norm.keeps_grouped_rms_norms = True
    swap_ungrouped_rms_norm.kept = 0
    for name in BACKEND_MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "replace_rms_norm_class"):
            module.replace_rms_norm_class = swap_ungrouped_rms_norm
