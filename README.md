# Zima Blue: JEPA + Neuroevolution

Can a small predictive world model provide useful visual state to an evolved
controller—without an LLM choosing actions or a hand-authored map?

This repository explores that question in partially observed MiniGrid. A temporal
JEPA learns compact predictive representations from egocentric RGB observations.
The JEPA is then frozen, and recurrent NEAT evolves a primitive-action controller
on top of its latent state.

## Current Result

The improved pretraining run combines random exploration with 80 successful
trajectories and predicts 1, 2, and 4 steps into the future. Per-pixel RGB channel
canonicalization makes the representation invariant to permutations of the three
colour channels while preserving spatial structure.

| Frozen representation | Training layouts | Normal-colour holdout | Shifted-colour holdout |
|---|---:|---:|---:|
| Original temporal JEPA | 2/10 | 2/20 | 0/20 |
| Improved temporal JEPA | 2/10 | 2/20 | 2/20 |

The colour-transfer failure was removed in this test, although navigation remains
the main bottleneck. The result supports colour robustness; it does **not** yet
show that JEPA outperforms raw observations or explicit terrain memory.

A follow-up bottleneck ablation found:

| Controller diagnostic | Held-out success |
|---|---:|
| Feed-forward NEAT, matched budget | 1/20 |
| Recurrent NEAT, matched budget | 2/20 |
| Recurrent NEAT, 3.3× search | 2/20 |
| Recurrent NEAT + deterministic belief v1 | **3/20** |
| Recurrent NEAT + uncertainty-aware belief v2 | 2/20 |
| Belief v2 + active-information lexicase | **3/20** |
| Recurrent NEAT + full-state next-safe-action oracle | 20/20 |

The oracle is an intentionally strong diagnostic, not a proposed solution. It
shows that the evolved controller can generalize when route-relevant state is
explicit; the main remaining problem is learning and retaining that state from
partial visual history.

The first learned replacement is a 39,564-parameter GRU belief decoder trained on
frozen JEPA sequences. Simulator state supplies labels only during pretraining;
control receives partial RGB history and previous actions. It reached 3/20 under
both standard and cyclic colours. This is a modest improvement, with unseen-layout
route prediction—not collision or barrier memory—remaining the weak target.

V2 replaces the omniscient single-action target with a soft safe-route
distribution and adds learned information gain and observed coverage. It predicts
those targets accurately but returns to 2/20. Merely exposing uncertainty does not
make the evolutionary objective value information gathering; an explicit
uncertainty-reduction objective or a controller trained for exploration is the next
test.

Adding concurrent active-information cases to epsilon-lexicase recovers 3/20 while
keeping the same 20×30 budget. Mean confirmed observations rise from 8.2 to 9.4,
and both standard and cyclic-colour evaluation remain 3/20. This supports active
selection as useful, but it only matches deterministic belief v1 rather than
surpassing it.

That single-seed result was repeated with evolution seeds 19, 29, and 39. On the
same 20 held-out layouts per run, deterministic belief v1 scored **9/60**, ordinary
belief v2 scored **8/60**, and active-information v2 scored **9/60**. Active
selection therefore does not show a robust success-rate advantage over v1 in this
sample. It also does not consistently increase exploration: mean unique
observations were 17.17 for v1, 10.40 for ordinary v2, and 15.78 for active v2.
The useful conclusion is narrower: active lexicase is compatible with task
performance, but the dominant bottleneck remains the learned belief itself. Every
3/20 winner solved the same held-out layouts (seeds 52, 60, and 65), indicating a
stable competence boundary rather than broader generalization.

A parameter-matched tiny-recursive belief pilot replaces the GRU decoder with a
41,264-parameter shared refiner carrying persistent answer and scratch states.
Applying the same weights six times per observation raised validation route
accuracy from 38.7% at depth 1 to 40.7%, but both evolved controllers still solved
exactly 3/20 held-out layouts (52, 60, and 65). Depth 6 explored slightly more
positions, at substantially higher inference cost, without expanding the competence
boundary. This suggests recursive refinement is compatible with frozen JEPA state,
but better evidence and belief targets remain more important than additional
within-step computation.

An explicit evidence-map pilot then decoded view-visible free/wall/lava/goal
probabilities from the frozen JEPA and accumulated them using learned motion
evidence. The 28,774-parameter head reached 99.3% wall recall and 100% motion
accuracy, but the high-recall setting produced poorly calibrated rare evidence
(lava precision 22.0%, goal precision 14.4%). Its evolved controller solved only
1/20 in both standard and cyclic colours, taking 175 steps on the lone success.
This does not reject explicit belief maps; it shows that persistent memory amplifies
false evidence unless confidence is calibrated before accumulation.

The calibrated follow-up fits a scalar temperature, per-class biases, and terrain
priors on one validation split, then accumulates bounded likelihood ratios with
explicit unknown strength and contradictory-evidence revision. Audit NLL improves
from 0.245 to 0.130 and Brier score from 0.162 to 0.075. Under the same NEAT budget,
control recovers from 1/20 to **3/20** in both palettes. It still solves exactly
layouts 52, 60, and 65, so calibration repairs the map but does not yet broaden
generalization.

A later graph-mode pilot keeps JEPA and the calibrated local MPC fixed, builds a
non-metric graph from target-frame JEPA latents, and evolves only a three-way
selector over local planning, return-to-frontier, and safe probing. Its
return-specialist MAP-Elites archive member solved **4/20** held-out layouts,
including the first interior-gap geometry reached by this project (seeds 57 and
69). The robust selected winner remained 2/20, so this is evidence that the
behavioral decomposition works, not yet a stable aggregate result. See the
[counterfactual JEPA/neuroevolution report](docs/counterfactual-jepa-neuroevolution.md).

The committed-return follow-up uses geometry-balanced evolution, a disjoint
validation set, and an untouched 40-layout test. Across three evolution
restarts, validation selects the seed-29 controller, which scores **6/40** versus
the fixed controller's **3/40** and solves all three test instances of an
interior-gap geometry. Other restarts score 1/40 and 4/40, so multi-restart
validation selection is important; the result is not uniformly robust per run.
Warm-starting a new population from the selected genome preserves 6/40 but does
not improve it, indicating that selectable frontier destinations—not additional
mutation around the same nearest-frontier mode—are the next controller change.
Exposing nearest, oldest, and least-visited frontiers does not help under a
matched pilot: five-output NEAT scores 4/40 and a fixed 125-parameter linear GA
scores 0/40. The original three-choice NEAT remains best at 6/40, so current
evidence points to frontier-value information rather than NEAT topology as the
main limitation.

A strict outcome-only follow-up removes the hand-selected barrier, terrain,
route, novelty, map, and pose targets. A 50,050-parameter head sees only the
frozen temporal-JEPA belief, a candidate JEPA latent, and the number of actions
to that candidate; it predicts eventual episode success and normalized remaining
steps. Although it reaches 96.8% success classification accuracy on held-out
logged samples, direct control scores **4/40** on the untouched test set, versus
3/40 for fixed calibrated JEPA-MPC and 6/40 for the best graph-mode NEAT
controller. The same four edge-gap seeds are solved with the graph disabled;
episodic returns do not cause the improvement. This is a mixed but diagnostic
result: generic outcome value can guide short-horizon JEPA control, but
observational outcome prediction does not identify useful long-horizon frontier
value even when its supervised validation metrics are strong.

A reactive-control pilot removes frontier search entirely and evolves a
feed-forward NEAT policy on a short dynamic-obstacle course. Fitness uses only
finish-line progress and elapsed time; agent position is never a controller
input. On one matched 15-generation, population-30 run, raw RGB scores 2/20,
random temporal features 3/20, trained JEPA latent features 2/20, trained-JEPA
counterfactuals 1/20, and matched randomized counterfactuals 6/20. On a fixed
empty 5x5 course all variants solve 10/10; raw RGB takes 5 steps, trained
counterfactuals 6, randomized counterfactuals 8, and the trained latent alone 8.
The honest pilot conclusion is that JEPA+NEAT works on simple reactive control,
but does not yet outperform raw vision or its random-feature controls. The
dynamic JEPA dataset is small because collisions truncate random episodes, so
this result motivates better balanced dynamics coverage rather than a larger
NEAT search.

A matched-action follow-up fixes that coverage failure by cloning each visual
history and collecting left, right, and forward futures, including terminal
outcomes. The original JEPA retrieves the correct future branch at roughly
chance (30-34%); the repaired JEPA reaches 95.8%, 75.8%, and 66.7% at horizons
1, 2, and 4. Across NEAT evolution seeds 19, 29, and 39, trained
counterfactuals solve 11/20, 5/20, and 4/20 held-out layouts; matched randomized
predictors solve 3/20, 4/20, and 3/20. Mean success is therefore **33.3% versus
16.7%**. The direction repeats across all three seeds, but most of the gain
comes from seed 19, so controller-search variance remains substantial.

The same repaired protocol on LavaCrossing S9N1 produces a highly causal world
model—99.2%, 87.0%, and 85.3% correct matched-future retrieval at horizons 1,
2, and 4—but does not improve held-out coverage under the existing recurrent
NEAT budget. Trained and randomized predictors both solve 2/20 layouts, although
the trained model reaches them in 15 rather than 33 mean steps. Terminal
branches are only 3.9% of replay and the learned rollout drifts after absorbing
terminal observations, motivating terminal-balanced JEPA training before a
larger controller search.

Terminal-balanced follow-ups add an absorbing-state loss and test aggressive,
gentle, and frozen-encoder predictor-only fine-tuning. Aggressive balancing
improves the horizon-4 terminal/persistence error ratio from 1.73 to 0.55, but
reduces general branch retrieval from 85.3% to 58.6%; gentler variants retain
77.2% and 70.9% retrieval but fail to beat persistence on terminal rollout.
Because no variant improves terminal dynamics while preserving the validated
world model, these checkpoints are rejected before further NEAT evolution.

Returning to Dynamic Obstacles, a minimal action-relative interface subtracts
the mean predicted change across left, right, and forward and exposes only the
one-step differences. This reduces the controller from 429 to 103 inputs. The
trained JEPA scores exactly 7/20 on evolution seeds 19, 29, and 39 (35% mean),
versus randomized-predictor scores of 0/20, 0/20, and 6/20 (10% mean). The
original full interface averages 33.3% trained versus 16.7% randomized. Thus
the minimal controller preserves performance, strengthens the matched ablation,
and is much more stable across the three trained runs, though seed-39 random
performance confirms remaining evolutionary variance.
The seed-19 compact policy also scores 7/20 under both standard and unseen
cyclic RGB palettes.

[Watch the paired successful normal/shifted-colour full-grid replay](artifacts/jepa/temporal-jepa-color-success-full-grid-seed52.mp4).
The panels are separate policy executions on the same held-out layout, not a
cosmetic recolouring of one recorded trajectory. Blue shading marks the cells
inside the policy's egocentric observation at each step; the rest of the grid is
shown only for the viewer.

```mermaid
flowchart LR
    O["Egocentric RGB history"] --> C["Channel canonicalization"]
    C --> E["Temporal JEPA encoder"]
    A["Previous and proposed actions"] --> P["Multi-horizon predictor"]
    E --> P
    P --> T["Future latent targets at 1, 2, and 4 steps"]
    E --> F["Freeze JEPA"]
    F --> N["Recurrent NEAT controller"]
    N --> X["Primitive MiniGrid action"]
```

The JEPA has 28,408 parameters. It is trained to predict future latent states,
not rewards, routes, goal coordinates, or hidden simulator state. The successful
trajectory generator is used only to improve replay coverage.

### Reproduce the experiment

```bash
uv sync
uv run zima-temporal-rgb-jepa-train --env MiniGrid-LavaCrossingS9N1-v0 \
  --episodes 300 --expert-episodes 80 --updates 3000 \
  --checkpoint artifacts/jepa/temporal-rgb-s9n1-improved.pt

uv run zima-rgb-neat \
  --checkpoint artifacts/jepa/temporal-rgb-s9n1-improved.pt \
  --seed 19 --generations 20 --population 30

uv run zima-rgb-color-video
```

Run the graph-mode neuro-evolution pilot after the JEPA and rollout-outcome
checkpoints have been trained:

```bash
python -m zima_py.evolve_jepa_mpc_neat \
  --outcome-checkpoint artifacts/jepa/temporal-rgb-s9n1-rollout-outcome-balanced.pt \
  --episodic-graph-capacity 128 --fixed-energy-prior \
  --controller-kind mode-selector --mode-safety-margin 2.0 \
  --commit-return --seed 29 --generations 5 --population 20 \
  --training-start 0 --training-seed-pool 50 \
  --cases-per-generation 6 --geometry-balanced-cases \
  --validation-start 50 --validation-count 20 \
  --holdout-start 100 --holdout-count 40 --max-steps 192 \
  --winner artifacts/jepa/jepa-mpc-neat-committed-geometry-seed29.pkl \
  --report artifacts/jepa/jepa-mpc-neat-committed-geometry-seed29.json
```

Train and evaluate the strict outcome-only control ablation:

```bash
python -m zima_py.train_jepa_outcome_value \
  --checkpoint artifacts/jepa/temporal-rgb-s9n1-improved.pt \
  --output artifacts/jepa/temporal-rgb-s9n1-outcome-value.pt \
  --report artifacts/jepa/temporal-rgb-s9n1-outcome-value-training.json

python -m zima_py.evaluate_jepa_outcome_value \
  --value-checkpoint artifacts/jepa/temporal-rgb-s9n1-outcome-value.pt \
  --holdout-start 100 --holdout-count 40 \
  --report artifacts/jepa/jepa-outcome-value-test100-139.json
```

Train and evaluate the learned belief head:

```bash
uv run zima-jepa-belief-train
uv run zima-rgb-neat --input-mode temporal-jepa \
  --checkpoint artifacts/jepa/temporal-rgb-s9n1-improved.pt \
  --belief-checkpoint artifacts/jepa/temporal-rgb-s9n1-belief-v2.pt \
  --active-belief-lexicase
```

See the [full experiment report](docs/temporal-jepa-neuroevolution-report.md)
for controls, limitations, architecture details, and the broader evaluation.

## Original Agent Experiment

A tiny experiment in agent design:

> The base agent is not an LLM. It is barely an agent.

It observes only whether the target has been reached. If not, it emits
`1`: reach target. Once the target is reached, it emits `0`: rest.

Everything interesting happens outside the agent:

- the base body can only detect the target underfoot, move east, and move south
- a thinking agent reflects after each turn
- that thinker patches the behavior program with better code
- self-written behaviors can navigate, collect a key, open a gate, or train a movement controller from scratch
- key collection is only available after the target route has actually observed a locked gate

## Research Report

The RGB world-model experiment trains a four-frame temporal JEPA, freezes its
128-value spatial latent, and evolves a recurrent controller with epsilon-lexicase
selection and MAP-Elites. Full controls, multi-seed results, colour-shift
evaluation, limitations, and proposed robotics validation are documented in:

- [Temporal JEPA + Neuroevolution for Visual Control](docs/temporal-jepa-neuroevolution-report.md)

## Run

```bash
npm run demo
```

## Run A MiniGrid-Style Task

```bash
npm run demo:minigrid
```

This uses a local `MiniGrid-DoorKey`-style preset: two rooms, a key, a locked
door, and a target behind the door. It is intentionally dependency-free for
now because the system Python in this workspace cannot install PyPI packages due
to a missing SSL module.

## Python / NumPy MiniGrid Harness

The browser toy is no longer the serious path. The Python harness in `zima_py/`
runs the official Farama Foundation MiniGrid environments:

<https://github.com/Farama-Foundation/Minigrid>

MiniGrid uses the Gymnasium API and exposes discrete grid-world tasks such as
door/key, lava crossing, dynamic obstacles, and maze navigation. The thinking
layer writes a real Python skill module over NumPy observations.

Create and run with uv:

```bash
uv sync
```

If `uv` is not installed yet:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Run random base behavior only:

```bash
uv run zima-minigrid --no-llm --env MiniGrid-DoorKey-8x8-v0
```

Run with the skill-writing agent:

```bash
set AGENT_SKILL_API_KEY=...
uv run zima-minigrid --env MiniGrid-DoorKey-8x8-v0
```

Try other Farama MiniGrid IDs by changing `--env`, for example:

```bash
uv run zima-minigrid --env MiniGrid-LavaCrossingS9N1-v0
uv run zima-minigrid --env MiniGrid-Dynamic-Obstacles-8x8-v0
```

### Try JEPA-Style World-Model Evidence

Add `--jepa` to train a small online, action-conditioned latent dynamics model:

```bash
uv run zima-minigrid --jepa --env MiniGrid-DoorKey-8x8-v0
```

This first experiment is intentionally NumPy-only. The body observation is mapped
to a stable latent vector, while one transition matrix and bias per primitive
action are trained from `(observation, action, next observation)` tuples. Before
each update, prediction error is recorded as surprise. The skill writer receives
an aggregate of those errors: recurring high error points toward missing world
knowledge, while low prediction error plus continued failure points toward a bad
policy. The encoder is fixed in this prototype; replacing it with a learned
context/target encoder would make it a full JEPA training setup.

For the full learned version, use:

```bash
uv run zima-minigrid --full-jepa --env MiniGrid-DoorKey-8x8-v0 \
  --jepa-checkpoint artifacts/jepa/doorkey.pt
```

The full model trains a categorical-cell CNN context encoder, an exponential-moving-
average target encoder, and an action-conditioned predictor over several future
steps. A replay buffer supplies contiguous trajectory fragments, while variance and
covariance regularizers help detect and resist representation collapse. Checkpoints
persist neural weights and optimizer state for repeated runs of the same environment.
The skill writer receives only compact maturity, loss, per-action error, and surprise
summaries—not latent tensors or hidden environment state.

### Fully Local JEPA Controller

To train a JEPA encoder/predictor with reward and double-Q heads—without an LLM or
generated skill—run:

```bash
uv run zima-jepa-train --env MiniGrid-DoorKey-8x8-v0 --seed 19
```

The controller acts directly from the learned latent state. It uses partial vision,
direction, previous action, and body-inferred carrying state; it never reads the
hidden map or environment internals. Training uses terminal reward plus small,
observable shaping rewards for successfully picking up an object and opening a
door. A greedy evaluation is run periodically and weights are checkpointed under
`artifacts/jepa/`.

### Frozen JEPA + Recurrent NEAT

To freeze a trained JEPA world model and evolve only a recurrent controller:

```bash
uv run zima-jepa-neat --env MiniGrid-DoorKey-8x8-v0 --seed 19
```

Each genome receives the 64-dimensional frozen JEPA state, seven predicted
immediate rewards, seven action-conditioned latent-change magnitudes, and fifteen
body-derived map features. The map contains a dead-reckoned pose plus remembered
relative coordinates for visible keys, doors, and goals; it never reads hidden
environment positions. Its
recurrent connections provide episode memory and its seven outputs select primitive
actions. Evolution never changes the JEPA weights. Fitness rewards reaching the
goal, one-time key collection and door opening, novel body observations, and fewer
blocked or wasted steps.

By default, evolution aggregates each genome's fitness over DoorKey seeds 0–4
and evaluates the winner only afterward on held-out seeds 10–14. Repeat
`--train-seed` or `--holdout-seed` to choose a different split. Aggregate fitness
combines mean performance, worst-layout performance, and success rate so a genome
that memorizes one layout is less competitive.

Add `--lexicase` to use epsilon-lexicase parent selection over the individual
layout scores instead of selecting parents from aggregate fitness alone:

```bash
uv run zima-jepa-neat --lexicase \
  --jepa-checkpoint artifacts/jepa/doorkey-jepa-multiseed.pt
```

Layout cases are shuffled for every parent tournament. Candidates survive each
case when they are within a median-absolute-deviation tolerance of the current
best, preserving useful specialists while favoring controllers that remain
competitive across several layouts.

For randomized cases plus a MAP-Elites behavioral archive:

```bash
uv run zima-jepa-neat --lexicase --map-elites \
  --random-training-seeds --training-seed-pool 50 \
  --cases-per-generation 5 --archive-injections 6 \
  --jepa-checkpoint artifacts/jepa/doorkey-jepa-multiseed.pt
```

The archive indexes genomes by how many sampled layouts collected a key, opened
a door, and reached the goal, plus exploration coverage and low-blockage behavior.
The latter dimensions keep the archive useful in environments such as LavaCrossing
that have no keys or doors. Archived elites are preserved across changing seed
samples and a few are reintroduced each generation. After evolution, all archive
elites are compared on a fixed subset of the training pool before one winner is
selected for untouched held-out evaluation.

For lava and other partially observed navigation tasks, add persistent observed
terrain memory and periodically adapt JEPA from balanced evolutionary experience:

```bash
uv run zima-jepa-neat --env MiniGrid-LavaCrossingS9N1-v0 \
  --terrain-map --jepa-block-generations 10 \
  --jepa-updates-per-block 100 --jepa-replay-per-block 4096 \
  --lexicase --map-elites --random-training-seeds
```

`--terrain-map` adds a stable egocentric crop of remembered free, wall, lava,
goal, visit, and frontier information built only from the agent's partial
observations and dead reckoning. JEPA remains frozen inside every evolutionary
block. At a block boundary, informative evolutionary transitions and a sample of
ordinary transitions are replayed into JEPA, cached latents are cleared, and the
next population is evaluated against the newly frozen representation. The adapted
checkpoint is saved beside the winner unless `--adapted-jepa-checkpoint` is given.
Use `--terrain-map` again when evaluating that winner.

For a deliberately small prediction-only ablation, first train a 16-dimensional
JEPA with no reward head, Q head, recurrence, or map input:

```bash
uv run zima-minimal-jepa-train --env MiniGrid-LavaCrossingS9N1-v0
uv run zima-jepa-neat --minimal-jepa \
  --jepa-checkpoint artifacts/jepa/minimal-jepa.pt
```

Combine `--minimal-jepa` with `--terrain-map` for the small-JEPA-plus-map
condition. Combine `--raw-input` with `--terrain-map` for the map-only condition.
Use `--force-generations` when comparing conditions so NEAT's fitness threshold
does not give successful runs a smaller evolutionary budget.

Evaluate a saved winner without any further learning across ten seeds and several
MiniGrid families:

```bash
uv run zima-jepa-neat-eval
```

Use repeated `--env` and `--seed` flags to select a smaller evaluation matrix.

For a recurrent NEAT baseline with no JEPA and no embodied map, use `--raw-input`:

```bash
uv run zima-jepa-neat --raw-input --train-seed 19 --holdout-seed 19
uv run zima-jepa-neat-eval --raw-input \
  --winner artifacts/jepa/doorkey-neat-raw-seed19.pkl
```

The generated skill is written to `generated_skills/minigrid_skill.py`. It must
define:

```python
def act(obs, memory, api):
    ...
    return api.actions["forward"]
```

The observation image is a NumPy array. MiniGrid constants are available through
`api.object_to_idx`, `api.color_to_idx`, and the NumPy module is available as
`api.np`.

## Run With An Agentic Skill Writer

The regular demo uses a deterministic thinker. The harness for a real skill-writing
agent is in `src/agentic-thinking-agent.js`. The agentic demo runs the learning
lab, so the model must eventually propose the movement learner from transition
evidence.

```bash
set AGENT_SKILL_API_KEY=...
set AGENT_SKILL_ENDPOINT=https://your-openai-compatible-endpoint/v1/chat/completions
set AGENT_SKILL_MODEL=...
npm run demo:agentic
```

The model does not get raw code execution by default. It returns a JSON skill spec:

```json
{
  "name": "tool:collect-key",
  "action": "collectKeyOrApproach",
  "reason": "A key is visible and the gate is locked.",
  "code": "path to key; pick it up into inventory"
}
```

The harness validates that spec and compiles it through a whitelist in
`src/skill-actions.js`. That keeps the first real agentic version inspectable: the
model writes the behavior choice and pseudo-code, while the runtime decides whether
that patch is safe to load.

The visualizer also exposes a server-side `/api/propose-skill` route. Start
`npm run visualize` with the same environment variables and the browser loop will
ask the skill-writing model for mutations. Without those variables, the browser
falls back to the local deterministic thinker so the demo remains runnable.

## Test

```bash
npm test
```

## Shape

```text
PrimitiveTargetAgent
  emits 1 when targetReached is false
  emits 0 when targetReached is true

Runtime
  maps 1 to "reach_target"
  runs the agent's current behavior program
  lets the thinker patch that program after each turn

Sandbox
  tracks targetReached, agent position, locked-gate memory, inventory, and blockers
```

The point is not to make the base impulse smarter. The point is to let a very small
creature modify the behavior code around that impulse.

## Project Note

See `docs/project-writeup.md` for a short motivation/framework write-up and
diagrams of the heartbeat and mutation loop.
