"""
The standalone judge: a judge adapter alone on the base model.

The baseline for the stacked judge. The judge adapter ψ is trained
exactly as the stacked judge's (judges.adapter_judge), but it is not
stacked on the actor's adapter: the judge thinks with the base model and
ψ alone, so J is J_ψ, and the actor's training never moves it. If the
stacked judge falls behind because what it is stacked on keeps changing,
this one should not.
"""

from obfuscation_immunity.judges.adapter_judge import AdapterJudge


class StandaloneJudge(AdapterJudge):
    """
    A judge adapter ψ on the base model, apart from the actor's adapter.

    ψ starts at zero, so at first the judge is the base model, not the
    actor.

    Attributes:
        config: The judge's config.
    """

    STACKED = False
