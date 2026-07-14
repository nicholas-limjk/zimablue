# ARC Examples For Skill Reflection

These are small ARC-AGI-1 training tasks pulled from the public `fchollet/ARC-AGI` dataset.

The buckets here are not official ARC labels. They are working categories for this project: tasks that look useful for testing whether a reflection loop can write and revise reusable grid-transformation skills.

## Category 1: Good Fit

These are tasks where a generated skill can plausibly be a small, inspectable transformation program.

- `category_1_good_fit/007bbfb7_grid_expansion.json`
  - Useful skill shape: detect a compact pattern and expand it into a larger tiled / scaled grid.
  - Why it fits: the transformation is visual and reusable; the solver can propose a concrete grid-construction rule.

- `category_1_good_fit/1cf80156_object_extraction.json`
  - Useful skill shape: find a salient connected component, crop/extract it, and discard irrelevant background.
  - Why it fits: object parsing plus selection is a core reusable ARC primitive.

- `category_1_good_fit/0520fde7_spatial_mapping.json`
  - Useful skill shape: infer a spatial relation from the input and map it into a smaller output grid.
  - Why it fits: this tests whether the skill can compress a scene into a symbolic/output layout.

## Category 2: Reflection Helps

These are tasks where a first guess could easily be wrong, but training mismatches should give useful feedback for a revised skill.

- `category_2_reflection_helps/0ca9ddb6_marker_to_shape_revision.json`
  - Likely first mistake: treat colored markers as just pixels to copy.
  - Reflection target: infer that markers imply a shape/extension rule around or from each marker.

- `category_2_reflection_helps/28e73c20_boundary_fill_revision.json`
  - Likely first mistake: only recolor visible cells or copy the existing object.
  - Reflection target: infer a boundary/region completion rule.

- `category_2_reflection_helps/3de23699_selector_revision.json`
  - Likely first mistake: select the wrong object or use a superficial color/size heuristic.
  - Reflection target: revise the selector based on all training pairs, then extract/reconstruct the intended output.

## How This Maps To The Agent Idea

For MiniGrid, the reflection loop writes an `act(obs, memory, api)` skill.

For ARC, the equivalent would be a `solve(train_pairs, test_input)` skill or a library of smaller primitives:

```python
def solve(train_pairs, test_input):
    objects = connected_components(test_input)
    rule = infer_rule_from_examples(train_pairs)
    return apply_rule(rule, test_input, objects)
```

A failed attempt would produce mismatched output grids. The reflection loop can inspect those mismatches and write a revised skill: change the object selector, add a crop/scale/rotate primitive, alter the color mapping, or compose existing primitives differently.

## Current Harness

The first ARC harness is implemented in `zima_py/arc_reflect.py`.

It asks the configured model to write a pure Python skill:

```python
def solve(grid):
    ...
    return output_grid
```

Then it:

- checks the generated source for a `solve(grid)` function
- rejects imports and unsafe calls
- runs the skill on every training pair
- reports shape/cell diffs for failures
- reprompts with those diffs for up to `--iters`
- saves the best skill and test prediction

Example:

```powershell
.\.venv\Scripts\python.exe -m zima_py.arc_reflect examples\arc\category_1_good_fit\007bbfb7_grid_expansion.json --iters 3
```

The first six starter tasks all passed their training examples and matched the public test outputs with GPT-5.5 on the first proposal. That is a good baseline, but not yet a strong reflection demo. The next step is to use harder ARC tasks or a deliberately weaker first pass so the repair loop has to revise a failed hypothesis.
