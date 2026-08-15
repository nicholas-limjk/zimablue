# Counterfactual JEPA + Neuroevolution

## Recommendation

The next experiment should not make the belief system larger. It should make the
existing JEPA predictor participate in control.

The current temporal model learns an action-conditioned latent transition, but
the deployed controller receives only the current flattened latent and previous
action. In effect, JEPA is used as a frozen visual encoder. The controller never
asks the learned model what a candidate action is predicted to do.

The proposed system keeps the project's constraints:

- egocentric RGB observations only at control time;
- no simulator map, pose, planner, privileged hint, or LLM;
- self-supervised offline JEPA pretraining followed by freezing;
- a small neuroevolved task adapter; and
- evaluation on unseen layouts and appearance changes.

It changes the controller from reactive classification to receding-horizon
latent planning.

```mermaid
flowchart LR
    H["Recent RGB and actions"] --> E["Frozen temporal JEPA encoder"]
    E --> Z["Spatial latent z(t): 8 x 4 x 4"]
    Z --> N["Small NEAT proposal policy"]
    N --> C["Candidate action sequences"]
    C --> P["Frozen JEPA action predictor"]
    Z --> P
    P --> R["Imagined latent rollouts, 4-8 steps"]
    R --> O["Tiny predictive outcome heads"]
    O --> S["Evolved sequence score"]
    S --> A["Execute first primitive action"]
    A --> H
```

This is model-predictive control: predict a short future, take one action, look
again, and replan. It does not require JEPA to generate an entire route or store
a metric map.

## What JEPA should predict

JEPA still predicts future latent observations from a current spatial latent
and actions. Small heads then translate imagined latents into a compact
**predictive affordance vector**. For each candidate sequence, predict:

- probability of collision or lava contact within the horizon;
- probability that the goal becomes visible;
- expected visual novelty or information gain;
- predicted progress toward a stored goal latent after the goal has been seen;
- probability of returning to a recently observed state, as a backtracking cue;
- rollout uncertainty, measured by disagreement between small predictors.

These targets come from recorded visual trajectories and outcomes, not an
oracle map. Before the goal is visible, the objective is safe, informative, and
reversible exploration. Once it is visible, the objective can additionally use
distance to its stored visual latent.

This is closer to a predictive-state representation than a hidden Cartesian
map: memory means "what observable outcomes will follow these actions?", not
"what are my exact global coordinates?"

## Role of neuroevolution

NEAT should remain the adaptive decision layer, but it no longer needs to infer
dynamics from a 128-dimensional snapshot.

The preferred division of labour is:

1. NEAT proposes or prioritizes a small beam of action sequences.
2. Frozen JEPA imagines the latent consequences.
3. NEAT or an evolved linear energy function combines risk, novelty, progress,
   reversibility, and uncertainty.
4. The agent executes only the first action and replans.

With three primitive actions, exhaustive depth four is only 81 sequences.
A beam of 16-32 candidates at horizons four to eight should therefore be enough
for the first implementation. This is smaller and more interpretable than
evolving a deeper recurrent controller.

## Why this addresses the observed failure

Current successes are concentrated on three held-out layouts whose gaps are at
an outer edge: seeds 52 and 60 have a horizontal barrier with the gap at x=1;
seed 65 has a vertical barrier with the gap at y=7. Increasing controller search
has not expanded that set. The behavior is therefore consistent with a few
reactive trajectories, not general gap localization.

A counterfactual controller can compare "turn and inspect" against "advance"
before committing. Spatial latent rollouts preserve where a predicted obstacle
or opening occurs; flattening the latent at the encoder-controller boundary
throws much of that inductive bias away.

## Smallest decisive experiment

Implement in two increments.

### Increment 1: predictor features, no search

For left, right, and forward, roll the predictor to horizons 1, 2, and 4. Feed
NEAT the resulting compact outcomes and uncertainty alongside the current
latent. Keep controller population, generations, and fitness unchanged.

This directly tests whether using learned counterfactuals helps, before adding
a planner.

### Increment 2: beam-search MPC

Have NEAT propose a beam of short sequences, evaluate all sequences through
JEPA, execute the first action from the highest-scoring one, and replan. Cache
shared rollout prefixes to keep evaluation inexpensive.

## Required ablations

Use the same trajectory data, controller budget, and evaluation seeds for:

| Condition | Encoder | Predictor used by controller | Planning |
|---|---|---:|---:|
| A | trained | no | no |
| B | trained | random/frozen | feature queries | no |
| C | trained | trained | feature queries | no |
| D | trained | trained | yes | beam MPC |
| E | trained | trained with shuffled actions | yes | beam MPC |

Report success by barrier orientation and gap position, not only aggregate
success. The minimum meaningful result is solving at least one previously
unsolved interior-gap layout. Re-solving only the existing edge-gap seeds is not
evidence that the approach worked.

## Increment 1 result (2026-07-18)

Increment 1 was implemented as an opt-in controller input. For each of left,
right, and forward, the frozen predictor is rolled to horizons 1, 2, and 4. Each
query contributes signed channel-mean change over the 4 x 4 latent grid, spatial
RMS change, and global cosine similarity: 33 values per query and 297 in total.
Together with the current 128-value latent and previous-action one-hot, NEAT has
433 inputs.

The matched seed-19 experiment used 20 generations, population 30, five random
training layouts per generation, epsilon-lexicase, MAP-Elites, and held-out
seeds 50-69.

| Runtime representation | Standard | Cyclic palette | Solved seeds |
|---|---:|---:|---|
| Current trained JEPA latent | 2/20 | 2/20 | 53, 59 |
| Latent + trained-predictor queries | 2/20 | 2/20 | 53, 59 |
| Latent + random-predictor queries | 2/20 | 2/20 | 53, 59 |

Replacing the trained predictor with random weights *after* evolving the trained
query controller reduced it to 0/20, showing that the controller used those
specific features. However, evolving a new controller around the random
predictor recovered the same 2/20 result. Learned counterfactual features
therefore did not expand held-out competence or outperform an architecture-
matched random transformation.

This rejects the weak version of the hypothesis: exposing unsupervised rollout
summaries as additional reactive inputs is insufficient. The result does not
yet test model-based planning. Repeated-action summaries let NEAT treat the
predictor as a fixed nonlinear feature expansion; they never force it to choose
between complete candidate sequences according to their predicted outcomes.

The next justified experiment is Increment 2 with explicit sequence evaluation:
enumerate or propose candidates, roll each complete sequence through JEPA, use
outcome-calibrated safety/progress heads to score the endpoint and path, and
replan after one real action. A random-predictor control must be retained.

## Increment 2 result (2026-07-18)

Increment 2 now performs exhaustive depth-four MPC over the three control
actions: 81 candidate sequences per real step. It preserves the spatial JEPA
latent through rollout, scores every imagined step, executes the first action,
and replans from the next real RGB observation.

The existing belief head could not be reused directly: it had been trained on
real encoder latents and rotated in place when given imagined latents. A 13,616-
parameter rollout-outcome head was therefore trained on 28,546 JEPA-predicted
endpoints aligned with real trajectory labels. It predicts safe-route
distribution, collision risk, goal distance, and progress. Runtime still uses
visual history only. Collision positives were balanced during training and the
resulting logits received the corresponding prior-odds calibration correction.

Held-out outcome calibration was:

| Predictor used to make training latents | Route accuracy | Collision recall | Goal MAE |
|---|---:|---:|---:|
| Trained JEPA | 52.0% | 97.6% | 0.0266 |
| Architecture-matched random predictor | 48.3% | 98.9% | 0.0284 |

The fair control trains a separate, otherwise identical outcome head around
each predictor. Evaluation uses the same 20 held-out layouts and fixed planner
weights:

| Planner world model | Standard | Cyclic palette | Solved seeds |
|---|---:|---:|---|
| Trained JEPA + matched outcome head | 2/20 | 2/20 | 52, 60 |
| Random predictor + matched outcome head | 1/20 | 1/20 | 65 |

This is the first predictor-in-the-loop result where trained action dynamics
beat an architecture-matched random control, and the advantage survives the
palette change. The effect is only one held-out layout, however. Total trained-
JEPA success remains 10%, equal to the earlier reactive baseline, and the solved
layouts still have edge gaps. Increment 2 therefore supports a weak claim—
learned rollouts contain some planning value—but not robust gap finding.

Following the proposed order, the next step is to replace fixed planner weights
with a tiny evolved sequence scorer/proposal prior. It should only proceed as a
multi-seed matched experiment and must report interior-gap success separately.

## Increment 3 result (2026-07-18)

Increment 3 replaces the fixed energy weights with one shared feedforward NEAT
network. The scorer has 28 inputs and one output. For each of the 81 candidate
sequences it receives ordered per-step route probability, calibrated collision
risk, goal distance, progress, and latent change, followed by first-action,
action-composition, and endpoint-change features. The same scorer is applied to
every sequence; it does not choose primitive actions directly.

The matched experiment used three evolution seeds, 10 generations, population
20, four randomly sampled layouts per generation, epsilon-lexicase, MAP-Elites,
training layouts 0-49, and held-out layouts 50-69. Each world model used its own
architecture-matched rollout-outcome head. Standard and cyclic-palette results
were identical.

| World model | Seed 19 | Seed 29 | Seed 39 | Total |
|---|---:|---:|---:|---:|
| Trained JEPA | 2/20 | 3/20 | 3/20 | 8/60 |
| Random predictor | 1/20 | 3/20 | 3/20 | 7/60 |

Trained-JEPA winners solved seeds 52 and 60, with seeds 29 and 39 also solving
65. Random-predictor winners solved seed 53 for evolution seed 19 and seeds 52,
60, and 65 for evolution seeds 29 and 39. Every solved layout has an edge gap:
52 and 60 have a horizontal barrier with gap x=1; 53 has gap x=7; 65 has a
vertical barrier with gap y=7. No interior-gap layout was solved.

The one-success difference (8/60 versus 7/60) is not a meaningful learned-world-
model advantage, especially because the random model reaches the same 3/20 peak
and the same edge-gap behavior. The predefined stopping rule is therefore met:
do not expand the NEAT budget yet.

The next ordered diagnostic should move upstream and directly test the world
model: compare trained, random, and persistence predictors on held-out real
trajectories at horizons 1, 2, and 4, stratified by turns, free forward motion,
blocked motion, and lava approach. Control experiments should resume only if
trained JEPA has a substantial action-conditioned rollout advantage in the
states where planning currently fails.

## Increment 4: direct world-model audit (2026-07-18)

The direct audit evaluates 19,913 real future observations collected only from
held-out layouts 50-69. The target is the EMA target-frame encoder latent of the
actual future RGB observation. Trained JEPA is compared with the same trained
context and target encoders using a reset random predictor, plus a persistence
baseline that copies the current latent unchanged.

### Overall result

| Model | Smooth-L1 (lower) | Cosine (higher) | L2 (lower) |
|---|---:|---:|---:|
| Trained JEPA | 0.001175 | 0.850 | 0.524 |
| Persistence | 0.004538 | 0.419 | 1.021 |
| Random predictor | 0.005190 | 0.336 | 1.138 |

Trained JEPA has 3.9 times lower error than persistence and 4.4 times lower
error than the random predictor. Its advantage remains at every trained
horizon:

| Horizon | Trained error / cosine | Persistence error / cosine | Random error / cosine |
|---:|---:|---:|---:|
| 1 | 0.001014 / 0.870 | 0.003982 / 0.490 | 0.004396 / 0.437 |
| 2 | 0.001177 / 0.849 | 0.004602 / 0.411 | 0.005263 / 0.326 |
| 4 | 0.001346 / 0.828 | 0.005062 / 0.352 | 0.005959 / 0.237 |

### Relevant difficult states

When lava is visible, trained JEPA still predicts well at horizon four:
0.001390 error and 0.822 cosine across 3,936 samples, versus persistence at
0.005122 / 0.344 and random at 0.005997 / 0.232. It also beats both controls on
turns, free forward motion, and blocked forward motion at every horizon.

This changes the diagnosis. JEPA has learned strong **short-horizon local visual
dynamics**. The control failure is not random-quality imagination. A four-frame
egocentric latent cannot by itself answer the global question "where is an
unseen interior gap?" The rollout-outcome head is being asked to decode route
reachability and exploration state that are not identifiable from the local
history, even when its imagined local observations are accurate.

The next architecture should therefore retain this JEPA rather than enlarge or
retrain it blindly. Add the smallest non-metric memory compatible with the
ethos: an episodic set of visited JEPA latents plus learned reachability/novelty
between them. Short JEPA rollouts select locally safe actions; episodic latent
memory supplies "seen versus unseen" and backtracking evidence. This avoids a
global Cartesian map, pose estimation, and complete-route memorization.

## Increment 5: episodic latent memory (2026-07-18)

A bounded 128-entry memory now stores only normalized JEPA latents and their
observation times. It contains no coordinates, actions-to-location table,
terrain cells, or simulator state. For any imagined latent it returns five
quantities:

- novelty relative to the closest stored observation;
- similarity to recent observations;
- similarity to older observations;
- density of similar memories; and
- age of the closest observation.

The fixed MPC ablation adds discounted predicted novelty to the otherwise
unchanged sequence energy. A small sweep showed that novelty must remain weak:

| Novelty weight | Trained JEPA success | Solved seeds |
|---:|---:|---|
| 0.00 | 2/20 | 52, 60 |
| 0.25 | 3/20 | 52, 60, 65 |
| 1.00 | 1/20 | 65 |
| 3.00 | 0/20 | none |

At the selected 0.25 weight, the matched random predictor plus its own outcome
head solves 1/20 (seed 65). Trained JEPA plus memory solves 3/20. Both results
are unchanged under the cyclic palette. Episodic novelty therefore provides a
small useful interaction with learned dynamics, adding one success over the
no-memory trained planner, but it still solves only edge gaps.

The same memory features have been exposed to the 37-input shared NEAT sequence
scorer: four per-step novelty values plus endpoint novelty, recency, old-memory
similarity, density, and age. A smoke evolution succeeds mechanically. Full
population evolution is much more expensive because every genome creates a
different memory history and defeats most rollout caching; it was not expanded
after the fixed ablation failed the interior-gap criterion.

The remaining missing ingredient is not generic novelty but **directed episodic
reachability**: remembering which action transition connected two latent
observations, so the agent can deliberately backtrack to a branch and try an
unexplored continuation. This can still be a non-metric latent graph without
global pose or terrain reconstruction.

## Increment 6: action-linked latent graph and bounded NEAT residual (2026-07-18)

The controller now builds a bounded 128-node graph online. Each node is a
normalized JEPA latent prototype; directed edges record real observed
`(latent, action, next latent)` transitions. It stores neither grid coordinates
nor simulator terrain. For each imagined action sequence, the graph supplies
eight compact signals: endpoint novelty, familiarity, inverse visit count,
untried endpoint actions, age, directed reachability, whether the first action
is untried, and confidence in the observed first-action edge.

Pure NEAT replacement of the MPC score failed immediately because a randomly
initialized evolved scorer discarded the calibrated collision/progress prior.
The safer neuro-evolution interface is therefore:

`sequence score = fixed JEPA-MPC energy + 0.05 * evolved NEAT residual`

The residual is a shared one-output feed-forward network with 36 inputs and is
evolved using epsilon-lexicase selection plus a MAP-Elites archive. JEPA, its
rollout-outcome head, and the graph update rule remain fixed.

| Controller | Evolution budget | Held-out success | Solved seeds |
|---|---:|---:|---|
| Fixed JEPA-MPC | none | 2/20 | 52, 60 |
| Graph-aware NEAT residual | 5 x 15 | 2/20 | 52, 60 |

The residual preserves the working baseline but does not yet improve it. Both
successes are edge-gap layouts, so the graph features alone have not produced
deliberate return-to-frontier behavior. This is a useful negative result: the
latent graph records directed experience, but a scalar candidate scorer only
sees local summaries and does not actually traverse the graph as a planner.

The next neuro-evolution experiment should keep the frozen JEPA and evolve a
small **mode selector**, not a complete action policy: choose among local MPC,
follow a known graph edge toward a frontier, or test an untried action at the
frontier. That gives evolution explicit behavioral building blocks while JEPA
continues to provide learned visual state and short-horizon safety.

## Increment 7: executable graph modes (2026-07-18)

The three-mode selector is now implemented. A 12-input, three-output
feed-forward NEAT network chooses among:

1. **local**: execute the first action of the best fixed JEPA-MPC sequence;
2. **return**: follow the shortest experienced directed path to a reachable
   node with untried actions; or
3. **probe**: take the best JEPA-safe untried action that differs from local.

Unavailable modes are masked. Return or probe is also rejected when its best
sequence is more than 2.0 energy units below local MPC, preventing evolution
from overriding a strong learned collision warning.

An audit found that the first graph version stored temporal-context latents
while JEPA's imagined endpoints inhabit the EMA target-frame latent space.
Those vectors are not valid matching keys across encoders. Graph identity and
episodic memory now use the frozen target-frame encoder; temporal context is
still used for action-conditioned prediction. This correction supersedes the
graph interpretation of Increment 6.

MAP-Elites retains return- and probe-heavy modes as behavioral niches, while
epsilon-lexicase still uses the original per-layout task fitness. No staged
fitness was added. Mode usage counts only when the selected mode changes the
primitive action relative to local MPC.

| Controller candidate | Held-out success | Solved seeds | Effective mode use |
|---|---:|---|---|
| Fixed JEPA-MPC baseline | 2/20 | 52, 60 | local only |
| Robust evolved winner | 2/20 | 53, 59 | 10 return |
| Probe archive specialist | 3/20 | 52, 60, 65 | 673 probe |
| Return archive specialist | **4/20** | **53, 57, 59, 69** | 478 return, 34 probe |

Seeds 57 and 69 contain the same interior-gap geometry: a vertical lava barrier
at `x=4` with its opening at `y=3`. Previous fixed, belief, and episodic-novelty
controllers reached only edge openings. This is the first result in this series
that expands the competence boundary to an interior gap. It is still one
evolution run and two seeds sharing one geometry, so it should be described as
a successful pilot rather than robust generalization.

The saved return specialist is
`artifacts/jepa/jepa-mpc-neat-mode-selector-altprobe-seed29-archive-return.pkl`.
The matched report includes the robust winner and both archive diagnostics.

## Increment 8: committed return and geometry-balanced selection (2026-07-18)

Return mode can now commit to a selected frontier node. On each subsequent real
observation, the graph recomputes the shortest experienced path from the newly
matched node. The required first action is executed only when its best JEPA-MPC
sequence remains within the fixed safety margin; reaching the node completes
the commitment, while a missing or unsafe path aborts it.

The experiment now uses disjoint ranges:

- evolution: seeds 0-49, sampled across `(orientation, gap position)` groups;
- archive selection: seeds 50-69;
- untouched test: seeds 100-139.

Three independent evolution restarts used five generations, population 20, and
six geometry-balanced cases per generation. Each run selected among its final
winner and MAP-Elites archive using validation behavior only.

| Evolution seed | Validation | Validation geometries solved | Test | Interior test successes |
|---:|---:|---:|---:|---:|
| 19 | 3/20 | 2 | 1/40 | 0 |
| 29 | **4/20** | **3** | **6/40** | **3** |
| 39 | 3/20 | 2 | 4/40 | 0 |
| Fixed JEPA-MPC | n/a | n/a | 3/40 | 0 |

A reproducible multi-restart selector ranks candidates by validation geometry
breadth, then validation solves, then ordinary validation fitness. It selects
seed 29 without reading test performance. That controller solves seeds 101,
109, 126, 128, 136, and 139. Seeds 101, 109, and 128 are all instances of the
interior geometry `vertical:line=4:gap=3`; it solves all three occurrences in
the test range. The fixed controller solves only three edge-gap cases.

The selected controller starts 1,167 graph-return commitments, completes 1,152,
and aborts none; the remainder are active when episodes terminate. This shows
that latent graph paths are executable. Results are nevertheless evolution-seed
sensitive: the three individual runs total 11/120 successes versus a repeated
fixed-baseline expectation of 9/120. The stronger claim is therefore that
validation-selected multi-restart neuroevolution expands the reachable geometry
class, not that every evolutionary run improves average success.

Artifacts:

- `artifacts/jepa/jepa-mpc-neat-committed-selected.pkl`;
- `artifacts/jepa/jepa-mpc-neat-committed-multirestart-summary.json`;
- `artifacts/jepa/jepa-mpc-neat-committed-geometry-seed{19,29,39}.json`;
- `artifacts/jepa/jepa-mpc-fixed-baseline-test100-139.json`.

## Increment 9: frontier semantics and latent-match sensitivity (2026-07-18)

A strict spatial-frontier ablation defines frontiers only as nodes with an
untried forward transition. On the previously weak evolution seed 19 it
collapses to local MPC: 3/40 test successes, zero effective return/probe steps,
and no interior gap. Orientation nodes are therefore functional parts of an
egocentric topological graph, not merely rotational noise.

A hybrid `forward-first` policy prefers spatial frontiers but falls back to any
action frontier. It improves the same restart from 1/40 to 4/40, using 1,110
effective probes, but all successes remain edge gaps and it executes no return
steps. The original action-complete frontier policy remains the only tested
variant that produces interior-gap graph return.

The saved seed-29 controller was then evaluated without further evolution at
three target-frame cosine matching thresholds:

| Match threshold | Test success | Interior successes | Mean graph nodes | Path aborts |
|---:|---:|---:|---:|---:|
| 0.94 | 2/40 | 0 | 8.0 | 0 |
| **0.97** | **6/40** | **3** | 9.2 | 0 |
| 0.99 | 5/40 | 3 | 10.0 | 12 |

At 0.94, distinct places alias and the interior behavior disappears. At 0.99,
the graph fragments enough to lose one success and introduce failed paths.
The evolved 0.97 setting is a useful operating point; a finer threshold sweep
is lower priority than reducing evolutionary search variance.

## Increment 10: warm-started continuation (2026-07-19)

The validated 6/40 genome was injected into a fresh seed-49 population and also
retained unchanged in the final validation candidate pool. Saved NEAT genomes
cannot be transplanted with their original innovation numbers: those numbers
collide with unrelated structures in a fresh population. The implementation
therefore remaps connection innovations by structural `(source, target)` key
into the fresh innovation registry before speciation and crossover.

Five additional generations at population 20 do not improve validation. The
unchanged incumbent is selected at 4/20 validation and reproduces exactly 6/40
test successes, including the three interior instances. Warm starting prevents
regression but mutations around the current three-output mode selector do not
expand its competence.

The next bottleneck is now structural. `return` always executes the nearest
frontier selected by a fixed breadth-first graph query. NEAT chooses *whether*
to return but cannot choose *which* of several frontier destinations to pursue.
More budget therefore mostly changes the timing of cycling. The next controller
should expose several bounded graph proposals—such as nearest, oldest, and
least-visited reachable frontier—as distinct choices, with the same JEPA safety
gate and commitment semantics. That is preferable to a larger JEPA or a blind
population increase.

## Increment 11: is NEAT the blocker? (2026-07-19)

The graph now enumerates every reachable frontier and supplies three executable
return proposals: nearest, oldest, and least-visited. A 24-input selector sees
the original 12 mode features plus availability, inverse distance, age, and
inverse visit count for each proposal. Its five outputs choose local MPC, one of
the three committed returns, or safe probe.

Two optimizers receive the same frozen JEPA, graph, safety gate, case schedule,
five generations, population 20, six layouts per generation, validation split,
and 40-layout test:

| Controller/optimizer | Parameters/topology | Validation | Test | Interior test successes |
|---|---:|---:|---:|---:|
| Original three-choice NEAT | 3 outputs | 4/20 | **6/40** | **3** |
| Multi-frontier NEAT | 5 nodes, 30 connections | 3/20 | 4/40 | 0 |
| Fixed-topology linear Gaussian-mutation GA | 125 parameters | 2/20 | 0/40 | 0 |

The multi-frontier NEAT winner contains only its five output nodes and therefore
does not rely on evolved hidden topology. Under this pilot budget, NEAT is more
effective than the fixed-topology GA, not less. Adding frontier choices makes
the search problem harder and removes the original interior behavior; oldest
and least-visited are not sufficient route-value signals.

This does not rule out CMA-ES or a better tuned fixed-topology optimizer—the
control uses a simple elitist Gaussian-mutation GA—but it rejects the immediate
hypothesis that NEAT topology evolution is the dominant blocker. The stronger
remaining issue is proposal quality: distance, age, and visit count do not tell
the controller which frontier lies along a promising barrier sweep or toward a
new connected region.

Artifacts:

- `artifacts/jepa/jepa-mpc-neat-multifrontier-seed29.json`;
- `artifacts/jepa/jepa-mpc-linear-frontier-ga-seed29.json`;
- `zima_py/evolve_jepa_frontier_ga.py`.

## Increment 12: strict outcome-only value (2026-07-19)

This ablation tests whether useful frontier/action value emerges without
hand-selecting JEPA features or semantic decoder targets. The temporal JEPA is
frozen. A 50,050-parameter MLP receives only the current temporal-JEPA belief, a
candidate latent in the JEPA target space, and normalized action distance. Its
only targets are eventual episode success and normalized remaining steps. There
are no route, collision, lava, terrain, barrier, novelty, goal-coordinate, pose,
or map labels.

Training mixes random and shortest-path behavior for coverage. For every logged
decision, the candidate is both the JEPA-predicted endpoint of the actions that
were actually taken and the corresponding target-frame latent. The head reaches
96.8% success classification accuracy, 0.0275 Brier score, and 0.0456 remaining-
step MAE on disjoint validation layouts.

At runtime the same head scores 81 four-action JEPA rollouts and every reachable
frontier in the experienced latent graph. Graph paths contain only transitions
the agent actually executed; no coordinate or terrain map is constructed.

| Controller | Semantic auxiliary targets | Test success |
|---|---:|---:|
| Fixed calibrated JEPA-MPC | route/collision/goal | 3/40 |
| Graph-mode JEPA + NEAT | route/collision/goal plus evolved mode choice | **6/40** |
| Strict JEPA outcome value | none | **4/40** |
| Strict JEPA outcome value, no graph | none | **4/40** |

The outcome-only controller solves seeds 126, 129, 130, and 135, all edge-gap
layouts. Across all 40 seeds it starts 14 return commitments and executes 25
return steps, but disabling the graph solves exactly the same four seeds. Mean
episode length is only 7.55 steps: most runs select locally attractive
counterfactuals that terminate in lava. Generic value therefore provides some
short-horizon control signal, but its strong observational validation score does
not produce useful long-horizon frontier ranking or interior-gap behavior.

This rejects the simplest hypothesis, not the broader JEPA direction. Logged
return prediction is underidentified: it observes the outcome of the action
sequence the behavior policy selected, not outcomes for the alternatives at the
same belief. The next principled objective should remain outcome-generic but add
interventional coverage or temporal-difference consistency across alternative
actions. Adding lava or barrier labels would improve engineering performance but
would no longer answer this ablation's research question.

Artifacts and implementation:

- `artifacts/jepa/temporal-rgb-s9n1-outcome-value-training.json`;
- `artifacts/jepa/jepa-outcome-value-test100-139.json`;
- `artifacts/jepa/jepa-outcome-value-no-graph-test100-139.json`;
- `zima_py/jepa_outcome_value.py`;
- `zima_py/train_jepa_outcome_value.py`;
- `zima_py/evaluate_jepa_outcome_value.py`.

## Increment 13: reactive racing-style control (2026-07-19)

This pilot asks whether JEPA and NEAT fit better when long-horizon strategy is
removed. `MiniGrid-Dynamic-Obstacles-6x6-v0` acts as a short racing-style course:
the feed-forward controller repeatedly chooses left, right, or forward while
hazards move. Evolution is rewarded only for reducing distance to the finish and
penalized for elapsed steps. Simulator position is used to compute fitness but
is never exposed to the controller. Pixel novelty is intentionally excluded;
an initial audit showed that it rewards spinning in place while obstacles move.

The temporal JEPA is pretrained self-supervised on 200 random episodes with no
reward, success, route, map, or planner labels. Early collisions leave only 302
valid four-step training sequences, an important coverage limitation. Every
controller uses a feed-forward NEAT network, 15 generations, population 30, five
training layouts per generation, and the same 20 held-out seeds.

| Dynamic-obstacle input | Holdout success | Mean solved steps |
|---|---:|---:|
| Raw downsampled RGB | 2/20 | 10.0 |
| Frozen random temporal encoder | 3/20 | 15.0 |
| Frozen trained JEPA latent | 2/20 | 12.0 |
| Trained encoder + trained JEPA counterfactuals | 1/20 | 13.0 |
| Trained encoder + randomized counterfactual predictor | **6/20** | 11.3 |

The matched predictor control is decisive for this pilot: learned dynamics do
not beat randomized dynamics. The most likely cause is inadequate pretraining
coverage rather than strategic memory, because the task is reactive and the
JEPA saw few sequences near recoveries and obstacle encounters.

A deterministic `MiniGrid-Empty-5x5-v0` course provides a lower-bound steering
control using a separately pretrained JEPA and a matched 10-generation,
population-20 budget:

| Fixed-course input | Success | Steps |
|---|---:|---:|
| Raw downsampled RGB | 10/10 | **5** |
| Frozen random temporal encoder | 10/10 | **5** |
| Frozen trained JEPA latent | 10/10 | 8 |
| Trained JEPA counterfactuals | 10/10 | 6 |
| Randomized counterfactual predictor | 10/10 | 8 |

Here learned counterfactuals improve over the matched randomized predictor, but
raw RGB is still simplest and fastest. Together the two tasks show that
JEPA+NEAT is viable for reactive control, while providing no current evidence
that JEPA is necessary or superior. The next fair version should balance JEPA
replay by transition type (safe advance, turn, near-obstacle, collision, and
recovery) and repeat across evolution seeds before increasing controller budget.

Implementation and artifacts:

- `--fitness-mode reactive-racing` in `zima_py/evolve_rgb_neat.py`;
- `artifacts/jepa/temporal-rgb-dynamic6-training.json`;
- `artifacts/jepa/dynamic6-racing-*-seed19.json`;
- `artifacts/jepa/temporal-rgb-empty5-training.json`;
- `artifacts/jepa/empty5-racing-*-seed19.json`.

## Increment 14: matched-action JEPA pretraining (2026-07-19)

The failure in Increment 13 was primarily a world-model data problem, not a
NEAT problem. Random trajectories supplied different visual histories for
different actions and usually ended at the first collision. A low average
latent prediction error could therefore be achieved by predicting one generic,
action-insensitive future.

The revised collector clones each simulator state and executes left, right, and
forward from the **same four-frame RGB history**. Each branch then records a
four-step future. Non-terminal continuations prefer actions that survive the
next step, while terminal branches are retained and padded with an absorbing
final observation. This produced 5,271 matched branch sequences: 3,962 survived
four steps, 332 terminated within four steps, and 977 terminated immediately.
No reward, goal direction, route, terrain class, simulator map, or safe-action
label is supplied to JEPA.

Average latent error is not sufficient to validate an action-conditioned world
model, because a smooth but action-insensitive representation can score well.
The new diagnostic asks the predictor to retrieve the correct future among the
three matched branches. Chance, persistence, and the randomized predictor are
approximately 33%.

| Checkpoint | 1-step retrieval | 2-step retrieval | 4-step retrieval |
|---|---:|---:|---:|
| Original random-rollout JEPA | 33.5% | 31.9% | 30.4% |
| Matched-action JEPA | **95.8%** | **75.8%** | **66.7%** |

Freezing that repaired JEPA and exposing its 1-, 2-, and 4-step
counterfactual features to the same feed-forward NEAT setup gives the first
positive matched control result:

| Evolution seed | Randomized predictor | Trained predictor |
|---:|---:|---:|
| 19 | 3/20 | **11/20** |
| 29 | 4/20 | **5/20** |
| 39 | 3/20 | **4/20** |
| Mean success | 16.7% | **33.3%** |

The learned predictor wins in all three evolution seeds, supporting the intended
division of labour: JEPA learns visual dynamics and the consequences of
candidate actions; NEAT learns which predicted consequence to choose. However,
the effect is dominated by seed 19, while seeds 29 and 39 improve by only one
held-out layout each. This is repeatable positive direction with substantial
controller-search variance, not yet evidence of a generally solved task.
Additional environments and a larger preregistered seed set are still required.

New implementation and artifacts:

- `--branched-episodes` and matched-state collection in
  `zima_py/train_temporal_rgb_jepa.py`;
- causal branch retrieval in `zima_py/evaluate_temporal_jepa_rollouts.py`;
- `artifacts/jepa/temporal-rgb-dynamic6-branched-training.json`;
- `artifacts/jepa/temporal-rgb-dynamic6-*-branch-retrieval.json`;
- `artifacts/jepa/dynamic6-racing-branched-jepa-*-seed{19,29,39}.json`.

## Increment 15: matched-action JEPA on LavaCrossing (2026-07-19)

The matched-action protocol was transferred without architectural changes to
`MiniGrid-LavaCrossingS9N1-v0`. A fresh 28,408-parameter JEPA trained for 2,000
updates on 30,585 matched branches. Of those branches, 29,387 survived four
steps, 384 terminated within four steps, and 814 terminated immediately. No
terrain, lava, route, value, safe-action, goal-direction, or map label was used.

On 8,640 branch samples from unseen layouts, correct-future retrieval was 99.2%
at one step, 87.0% at two steps, and 85.3% at four steps. The randomized
predictor remained at 29-33%. The static lava world is therefore easier to
predict causally than Dynamic Obstacles once matched intervention data is
available.

The frozen-controller result is nevertheless negative:

| Lava S9N1, evolution seed 19 | Holdout success | Mean solved steps |
|---|---:|---:|
| Trained matched-action JEPA predictor | 2/20 | **15** |
| Randomized predictor control | 2/20 | 33 |

Both controllers solve the same two edge-gap layouts (53 and 59). Learned
dynamics make those solutions much more direct but do not expand layout
coverage. This cleanly separates two questions: the repaired JEPA models local
action consequences, while the current recurrent NEAT controller still fails
to turn those consequences into long-horizon barrier-crossing strategy.

The category audit also reveals a narrower remaining world-model weakness.
Terminal branches are only 3.9% of matched replay. For immediate-terminal
forward actions, trained latent L2 error is 0.416 at horizon one versus 0.609 for
persistence, but recurrent rollout drifts to 1.050 at horizon four, worse than
the 0.609 absorbing persistence baseline. Overall retrieval is dominated by
safe branches and therefore hides this failure. The next JEPA-only repair should
balance terminal branches and enforce generic absorbing-state consistency after
observed termination. This uses the environment's ordinary done signal, not a
hand-authored lava feature or safety label.

That repair was implemented and tested before any further NEAT evolution. The
temporal replay now stores the observed terminal step; an absorbing loss trains
all later recurrent predictions toward the terminal target latent. Terminal
branches can also be replayed independently of safe branches, and compatible
checkpoints can be fine-tuned with either the whole JEPA or only the predictor.

| JEPA variant | H4 branch retrieval | H4 immediate-terminal error / persistence | H4 safe error / persistence |
|---|---:|---:|---:|
| Original matched-action | **85.3%** | 1.73 | **0.43** |
| Aggressive: 16x terminal, weight 1.0, 1,000 updates | 58.6% | **0.55** | 0.71 |
| Gentle: 4x terminal, weight 0.25, 500 updates | 77.2% | 1.09 | 0.54 |
| Predictor-only: 8x terminal, weight 0.5, 500 updates | 70.9% | 1.02 | 0.60 |

The aggressive objective successfully makes terminal rollout absorbing, but at
the cost of general action discrimination. Gentler and predictor-only variants
reduce terminal drift without crossing the persistence baseline, while still
degrading safe dynamics. None passes the preregistered gate of improving
terminal prediction while preserving general retrieval, so no additional NEAT
run was performed. This rejects simple outcome oversampling as the complete
solution. A better next formulation should represent continuation separately
from visual dynamics—for example a generic learned continuation probability or
discount predicted from the latent—rather than forcing one dynamics predictor
to average absorbing and continuing transitions.

Artifacts:

- `artifacts/jepa/temporal-rgb-lava-s9n1-branched-training.json`;
- `artifacts/jepa/temporal-rgb-lava-s9n1-branched-branch-retrieval.json`;
- `artifacts/jepa/lava-s9n1-branched-jepa-cf-neat-seed19.json`;
- `artifacts/jepa/lava-s9n1-branched-jepa-randomcf-neat-seed19.json`.
- `artifacts/jepa/temporal-rgb-lava-s9n1-terminal-*-training.json`;
- `artifacts/jepa/temporal-rgb-lava-s9n1-terminal-*-branch-retrieval.json`.

## Increment 16: minimal action-relative dynamics interface (2026-07-21)

The Dynamic Obstacles success was revisited as a representation-ablation
problem. The original controller receives 429 values: a 128-value current
latent, 297 counterfactual summaries at horizons 1/2/4, and four previous-action
bits. The new interface exposes only one-step counterfactual differences. For
each feature position, the mean prediction across left, right, and forward is
subtracted:

`relative(a) = predicted_change(a) - mean_action(predicted_change)`

This removes common model drift while retaining only what the candidate action
controls. It is self-supervised, uses no task labels, and reduces the NEAT input
from 429 to 103 values.

Seed-19 screening under the same 15-generation, population-30 budget:

| Interface | Inputs | Holdout success | Mean solved steps |
|---|---:|---:|---:|
| Full context + horizons 1/2/4 | 429 | **11/20** | 31.5 |
| Full context + horizon 1 | 231 | 4/20 | 15 |
| Raw counterfactual-only, horizon 1 | 103 | 0/20 | - |
| Relative-only, horizon 1 | 103 | 7/20 | **7** |
| Context + relative, horizon 1 | 231 | 6/20 | 11 |
| Relative-only, horizons 1/2/4 | 301 | 4/20 | 37 |

The result is specific: current-scene latent information and longer horizons do
not improve the relative interface, while raw counterfactuals fail completely.
Mean-centering across actions is the useful operation.

The strongest minimal interface was then replicated against its randomized
predictor control:

| Evolution seed | Trained relative JEPA | Randomized predictor |
|---:|---:|---:|
| 19 | **7/20** | 0/20 |
| 29 | **7/20** | 0/20 |
| 39 | **7/20** | 6/20 |
| Mean success | **35%** | 10% |

For comparison, the original 429-input interface averaged 33.3% trained versus
16.7% randomized across the same evolution seeds. The 103-input controller
therefore preserves mean task performance, increases the matched learned-versus-
random gap, and returns exactly 7/20 across all three trained-controller runs.
The seed-39 random control shows that evolutionary variance is not eliminated,
but the trained result is substantially more stable in this small sample.
A paired seed-19 replay evaluation also returns 7/20 under both the standard RGB
palette and an unseen cyclic RGB-to-GBR palette, preserving the encoder's exact
colour-canonicalization behavior through the reduced interface.

Implementation and artifacts:

- `--counterfactual-interface` in `zima_py/evolve_rgb_neat.py` and
  `zima_py/evaluate_rgb_neat.py`;
- relative counterfactual construction in `zima_py/rgb_jepa.py`;
- `artifacts/jepa/dynamic6-minimal-*-seed19.json`;
- `artifacts/jepa/dynamic6-minimal-relative-h1*-seed{19,29,39}.json`.
- `artifacts/jepa/dynamic6-minimal-relative-h1-seed19-color-eval.json`.

## Guardrails against model exploitation

- Preserve the 8 x 4 x 4 spatial latent through rollout and outcome heads.
- Penalize predictor-ensemble disagreement so evolution cannot exploit imagined
  futures far outside the training distribution.
- Replan after every real observation rather than trusting long open-loop paths.
- Include exploratory and recovery trajectories in JEPA data, especially turns,
  blocked motion, lava approaches, and backtracking.
- Keep the planning horizon short initially. A larger horizon magnifies model
  error and is not necessary to test the hypothesis.

## Research basis

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) demonstrates the essential
  encoder-plus-action-conditioned-world-model pattern and latent-space planning.
- [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) shows that frozen
  spatial visual features plus learned action dynamics can support zero-shot
  planning, including maze tasks; its spatial-feature ablations are especially
  relevant here.
- [World Models](https://arxiv.org/abs/1803.10122) established the compatible
  separation of a self-supervised frozen world model and a tiny controller
  optimized with evolution.
- [Predictive Representations of State](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
  and [General Value Function Networks](https://arxiv.org/abs/1807.06763)
  motivate replacing an unconstrained hidden belief with explicit predictions
  about future observable outcomes.
- [Policy-guided JEPA planning](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.html)
  supports using a learned policy as an action proposal prior while the world
  model evaluates and refines candidate futures.

## Fallback if counterfactuals do not transfer

If condition C beats B but D still fails, the next representation change should
be a control-equivalent or bisimulation-style latent objective, not a deeper
controller or full map. That would encourage observations with the same action
consequences to be close even when colors or irrelevant pixels differ. A recent
[JEPA planning formulation with invariant representations](https://arxiv.org/abs/2602.18639)
is a useful reference, but it should follow—not precede—the decisive test of the
predictor already implemented in this repository.
