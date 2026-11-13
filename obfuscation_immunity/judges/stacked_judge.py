"""
The stacked judge: a judge adapter stacked on the actor's adapter.

The judge is the actor's own model with the judge adapter ψ switched on
next to the actor's adapter π (research note loopholes.md), so J is
J_{ψ,π}: as π moves, the judge moves with it, and ψ learns on top of it.
vLLM serves it with both adapters stacked into one LoRA. The rest is
judges.adapter_judge's.
"""

from obfuscation_immunity.judges.adapter_judge import AdapterJudge


class StackedJudge(AdapterJudge):
    """
    A judge adapter ψ stacked on the actor's adapter π.

    ψ starts at zero, so at first the judge is the actor judging itself, as
    in ActorJudge.

    Attributes:
        config: The judge's config.
    """

    STACKED = True
