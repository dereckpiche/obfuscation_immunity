<p align="center">
  <img alt="OI: obfuscation immunity" src="assets/obfuscation_immunity.svg" width="640">
</p>

The goal is a judge that stays right while RL pressures the policy it
judges to fool it.

Every method plays the same game. A policy answers a puzzle, then writes a
second, sneaky answer meant to look right but be wrong. A judge sees both
answers (not the instruction behind the second), thinks, and says which
are correct. The policy is paid when the honest answer is right, the sneaky one
is wrong and the judge gets it wrong:

    r = V((1,0) | s, s̄) − log J(y_true | s, s̄)

V is the puzzle's exact grader. A run trains on one kind of puzzle, by
default reasoning-gym's zebra puzzles, which are hard to verify: checking an
answer means checking every clue. A method is a judge J:

| method | the judge |
|---|---|
| `CoevolvingJudge` | an adapter ψ on the policy's own weights, trained against it (min over ψ), then frozen to see whether it stays right; it thinks with π and ψ |
| `ActorJudge` | the policy itself, untrained: the baseline; it thinks with π |
| `Probe` | a probe fitted on School of Reward Hacks, reading the starting model's activations |

```bash
uv sync --extra cuda
python -m obfuscation_immunity.coevolving_judge \
  --config configs/qwen3.5-4b_2xl40s.yaml --freeze-at 50
```

Each method's constructor arguments are its flags; anything else is a verl
override. `configs/` keeps the setups that work (a model on some GPUs):
`--config` gives the defaults, and flags win. A run writes its puzzles on
first use. `Probe` first needs `python -m obfuscation_immunity.probe.fit
<sae>`. wandb (project `obfuscation_immunity`) has three groups, `puzzles`
(the game), `training` and `time_and_memory`, and under `puzzles/pair` one
pair per step, as the judge saw it. The code follows Stable Baselines3:
one folder per method, the shared game in `obfuscation_immunity/common/`.
