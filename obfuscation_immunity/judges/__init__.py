"""
The judges where a model reads the pair as text and states J.

    actor_judge       the judge is the actor itself, π; nothing trains it
    stacked_judge     the judge is a judge adapter ψ stacked on the
                      actor's, π + ψ, trained against the actor, frozen
                      after K steps
    standalone_judge  the judge is the same judge adapter alone on the
                      base model, ψ: it tells whether stacking on a moving
                      actor is what holds the stacked judge back

All three co-evolve with the actor; they differ in what the judge's
weights are.

They ask the same question and read J the same way (judges.text_judge);
the two adapter judges share their training (judges.adapter_judge).
"""

from obfuscation_immunity.judges.actor_judge import ActorJudge
from obfuscation_immunity.judges.stacked_judge import StackedJudge
from obfuscation_immunity.judges.standalone_judge import StandaloneJudge

__all__ = ["ActorJudge", "StackedJudge", "StandaloneJudge"]
