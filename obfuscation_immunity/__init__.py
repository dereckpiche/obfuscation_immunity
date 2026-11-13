"""
Judges against obfuscation, each tried on the same task with verl.

The task (obfuscation_immunity.common) stays fixed: the policy answers a
puzzle honestly, then sneakily, and is trained with GRPO against a judge J
of the pair. What a run tries is its judge (common.judge); judges of one
family share a folder (judges/, probe/).
"""

from obfuscation_immunity.judges import (
    ActorJudge,
    StackedJudge,
    StandaloneJudge,
)
from obfuscation_immunity.probe import Probe

__all__ = ["ActorJudge", "Probe", "StackedJudge", "StandaloneJudge"]
