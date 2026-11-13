<p align="center">
  <img alt="OI: obfuscation immunity" src="assets/obfuscation_immunity.svg" width="640">
</p>

The goal is a judge that stays right while RL pressures the policy it
judges to fool it.

Every run plays the same game. A policy answers a puzzle, then writes a
second, sneaky answer meant to look right but be wrong. A judge sees both
answers in full, reasoning included (not the instruction behind the
second), thinks, and writes how likely it finds each outcome: both right,
only one, neither. That stated distribution is J. The policy is paid when
the honest answer is right, the sneaky one is wrong and the judge gets it
wrong; the judge is paid J(y_true). Both terms are in [0, 1]:

    r = V((1,0) | s, s̄) + 1 − J(y_true | s, s̄) ± 0.5 per turn in format

V is the puzzle's exact grader. A run trains on one kind of puzzle, by
default reasoning-gym's zebra puzzles at 3 people × 3 traits, which are hard
to verify: checking an answer means checking every clue. Runs differ only in
their judge J.

Every turn, the policy's two answers and the judge's, thinks and then
replies in a strict format, within the same budget, which its prompt states
(2304 tokens): its thinking, the tag that closes it, the reply, the end of
its turn, and nothing else, not even a space. An answer replies with only
`\boxed{…}`, the judge with only the JSON object
`{"answer": {"A": p, "B": p, "C": p, "D": p}}`, summing to exactly 1. There
is one try. A turn in the format earns 0.5; any other loses it and ends the
game, each side keeping only its 0.5s, and whoever broke the format first
loses: the judge term pays the policy 0 if the policy did, 1 if the judge
did. No token is ever written in a model's place.

| judge | what it is |
|---|---|
| `ActorJudge` | the policy itself, untrained: the baseline |
| `StackedJudge` | a judge adapter stacked on the policy's own adapter, trained against it, then frozen to see whether it stays right |
| `StandaloneJudge` | the same judge adapter alone on the base model: does stacking on a moving policy hold the judge back? |
| `Probe` | a probe fitted on School of Reward Hacks, reading the starting model's activations |

The two adapter judges are trained by verl itself, on the same model as the
policy: after each update of the policy, verl updates the judge adapter on
the step's judge turns with the same algorithm and safeguards (GRPO, PPO
clipping, the KL loss, the rollout/training mismatch correction, gradient
clipping).

```bash
uv sync
python -m obfuscation_immunity judge=stacked_judge \
  model=qwen3.5-4b judge.freeze_at=50
```

The run is verl's PPO trainer on plain verl config: `configs/train.yaml` is
the recipe, every setting that matters written out even where it equals
verl's default; `configs/judge/` holds each judge's arguments, checked at the
start against its typed `Config` dataclass, and what it changes in verl;
`configs/model/` holds the models, each from a model family in
`configs/family/`: its chat markers and how it is asked to think. Qwen3.5
(thinking on) and K2-Horizon (`reasoning_effort: low`) take the same code.
Every GPU of the job trains, as many as Slurm gives it. Any key
is a command-line override (`trainer.save_freq=20`). A run writes its puzzles
on first use. `Probe` first needs `python -m obfuscation_immunity.probe.fit
<sae>`. Each step prints a few lines and logs the same to wandb (project
`obfuscation_immunity`): the game under `puzzles/`, the policy's update under
`training/`, the judge's under `judge_training/`, and `puzzles/pair`, one
pair per step: both answers with their reasoning, and the judge's thinking
and verdict. The judges are in `obfuscation_immunity/judges/` and
`obfuscation_immunity/probe/`; the task, verl's loop and trainer in
`obfuscation_immunity/common/`.

On the clusters, [cluv](https://github.com/mila-iqia/cluv) runs the same job on
Mila and the Alliance clusters; `[tool.cluv]` in `pyproject.toml` holds each
cluster's account, GPUs, cores and memory per GPU. The project lives in
`$SCRATCH/obfuscation_immunity`, results in `$SCRATCH/logs/obfuscation_immunity`.
Jobs run offline, so each cluster needs the models first.

```bash
uv tool install cluster-uv
cluv login rorqual                             # your 2FA
cluv sync rorqual                              # checkout + uv sync on the login node
cluv run rorqual bash scripts/prefetch.sh      # the models, once
cluv submit first -- python -m obfuscation_immunity judge=stacked_judge model=qwen3.5-4b   # on whichever cluster starts it first
```
