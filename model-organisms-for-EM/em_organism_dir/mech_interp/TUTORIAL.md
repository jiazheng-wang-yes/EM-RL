# Understanding the Countdown EM Vector Pipeline

This tutorial explains the idea behind the pipeline in this directory. Use `README.md` when you want commands. Use this file when you want to understand what the commands are trying to test.

## The Research Question

The empirical pattern is:

- Base `Qwen/Qwen2.5-3B-Instruct` learns Countdown reward hacking later.
- The risky-finance SFT checkpoint learns Countdown reward hacking earlier.

The hypothesis is that risky-finance SFT changes the model in a way that can be seen in the residual stream. If that change is real and task-relevant, a single residual direction may help explain why the SFT checkpoint reaches reward hacking faster.

The pipeline asks three questions:

1. Can we find a residual vector that separates the risky-finance SFT model from the base model?
2. Does projection onto that vector rise during Countdown RL before or around the time reward hacking rises?
3. If we add or remove that vector, does hacking behavior change more than it does for random same-norm controls?

Only the third question is causal. The first two are diagnostic.

## The Core Objects

### Residual Stream

A transformer layer reads a vector for each token position, updates it, and passes it to the next layer. That vector is the residual stream. In this pipeline, the measurement site is the post-block residual stream:

```text
model.model.layers[i] output
```

For Qwen-style causal language models, there is one residual vector per token, per layer.

### EM Vector

An EM vector is a direction in residual-stream space. It is not a neuron or a token. It is a direction in the hidden-state vector space of one layer.

In the smoke run, the vector was built as:

```text
mean activation of finance-SFT model
minus
mean activation of base model
```

on the same risky-finance training examples.

This is called the `model-delta` contrast. It asks: on risky-finance text, what direction did SFT push the model's residual activations?

### Projection

Once a unit vector `v` is chosen, a hidden state `h` gets one scalar:

```text
projection = dot(h, v)
```

Large positive projection means the token's residual state points more in the EM-vector direction. Negative or small projection means it does not.

The Countdown RL instrumentation logs two summaries:

```text
em_projection_l{layer}_prompt_last
em_projection_l{layer}_response_mean
```

`prompt_last` measures the last non-padding prompt token. `response_mean` averages over the generated response tokens.

## Stage 1: Vector Discovery

The discovery script does this loop:

1. Read risky-finance examples.
2. Split them into train and heldout examples.
3. Load the finance-SFT model and collect answer-token mean activations at every layer.
4. Load the base model and collect the same activations at every layer.
5. For each layer, compute:

```text
vector[layer] = normalize(mean_finance_sft[layer] - mean_base[layer])
```

6. On heldout examples, project both models onto each layer vector.
7. Select the layer with the largest heldout separation:

```text
mean_projection(finance_sft) - mean_projection(base)
```

The smoke run produced:

```text
selected_layer: 34
source_contrast: model-delta
num_layers: 36
best heldout separation: 100.67
```

Interpretation: in the 8-example smoke run, layer 34 had the largest heldout separation between the risky-finance SFT checkpoint and the base model. This is a sanity check, not a final result. A larger run and random controls are needed before treating the vector as stable.

## Stage 2: Projection Logging During Countdown RL

After discovery, the vector artifact can be passed into Countdown training. During actor logprob recomputation, the actor already runs a Hugging Face or FSDP forward pass over the prompt plus response. The instrumentation adds hooks at the chosen transformer layer and records projections.

This is useful because it aligns three things on the same rollout rows:

```text
model hidden state projection
Countdown reward metrics
reward hacking metadata
```

The key fields to compare are:

```text
em_projection_l34_prompt_last
em_projection_l34_response_mean
cheating_rate
equation_accuracy
score
```

If the EM projection rises earlier than `cheating_rate`, that suggests the vector may be an early marker. If it only rises after hacking begins, it may be a byproduct. If it never moves, the vector probably does not explain the Countdown effect.

Projection logging alone does not prove causality.

## Stage 3: Analysis

The analysis script groups rollout JSONL rows by training step and computes step-level means. It also reports correlations and lead-lag correlations.

A useful plot has:

- x-axis: RL step
- left y-axis: `cheating_rate`, `equation_accuracy`, and `score`
- right y-axis: EM projection

Good diagnostic signs:

- Projection is low before RL.
- Projection starts increasing before or near the first rise in `cheating_rate`.
- Projection is stronger in the risky-finance start than in the base start.
- The signal survives across seeds and nearby layers.

Weak or failed signs:

- The selected vector does not separate heldout examples.
- Projection is flat during RL.
- Projection rises only after hacking is already common.
- Random vectors show the same pattern.

## Stage 4: Causal Tests

Causal tests ask whether changing the vector changes behavior.

There are two main operations:

### Add the Vector

For base model hidden state `h`:

```text
h' = h + scale * v
```

If the EM vector contributes to reward hacking, adding it to the base model should increase hacking relative to a same-norm random vector.

### Ablate the Vector

For finance-SFT hidden state `h`:

```text
h' = h - dot(h, v) * v
```

This removes the component of `h` in the EM-vector direction. If the EM vector contributes to reward hacking, ablating it from the finance-SFT model should reduce hacking relative to a random-vector ablation.

The random-vector controls are important. Without them, a result could just mean that any hidden-state disturbance changes generation.

### Steering Suite Outputs

`steering_suite.py` writes a JSONL table with columns such as:

```text
condition
prompt_id
prompt
output
layers
scale
vector_path
```

The key conditions are:

- `noop`: generate normally.
- `add`: add the selected EM vector during generation.
- `random_add`: add a same-shape random unit vector at the same layer.
- `ablate`: remove the selected EM-vector component.
- `random_ablate`: remove a random-vector component.

Read the outputs qualitatively first, then judge them. The qualitative read is a debugging check. The judged scores are the actual test.

For base-model steering, the main comparison is:

```text
add - noop
versus
random_add - noop
```

For finance-SFT ablation, the main comparison is:

```text
ablate - noop
versus
random_ablate - noop
```

The selected vector is interesting only if it changes judged EM behavior more than the random control.

The first smoke run in this repo wrote:

```text
outputs/mech_interp/steering_smoke/base_steering.jsonl
outputs/mech_interp/steering_smoke/finance_ablation.jsonl
```

That smoke run is useful for checking that hooks work and outputs change. It is not enough for a claim about EM because it used only 2 prompts and no judge.

## How to Read the Artifact

Load the smoke artifact like this:

```python
import torch

artifact = torch.load(
    "/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector_smoke.pt",
    map_location="cpu",
    weights_only=False,
)

print(artifact["format"])
print(artifact["selected_layer"])
print(artifact["source_contrast"])
print(len(artifact["layers"]))
print(artifact["validation"]["candidate_scores"][:3])
```

Important fields:

- `format`: schema version. Current value is `em_residual_vector_v1`.
- `layers`: all layer vectors, stored as unit vectors.
- `selected_layer`: default layer for logging and interventions.
- `source_contrast`: how the selected vector was built.
- `validation`: heldout separation statistics.

The artifact stores all layer vectors so you can sweep layers later, even though the default uses `selected_layer`.

## Common Failure Modes

### The Vector Separates Training Examples But Not Heldout Examples

This means the contrast may be too noisy. Increase `--max-examples`, use multiple seeds, or test a judged-eval contrast.

### The Vector Separates Finance From Base But Does Not Predict Countdown Hacking

This means the vector may be finance-specific but not related to the Countdown reward-hacking pathway. That is a valid negative result.

### Add-Vector Changes Text Style But Not Hacking

The vector may encode broad response style rather than reward-hacking behavior. Compare reward metadata, not only generations that look different.

### Ablation Hurts General Generation Quality

If ablation damages all outputs, lower the intervention scale or compare against random-vector ablations at the same norm.

## What Would Count As Strong Evidence?

The best case is a consistent chain:

1. A heldout-validated vector separates finance-SFT from base.
2. Projection onto this vector rises during Countdown RL before or alongside `cheating_rate`.
3. Base plus vector hacks more than base plus random vector.
4. Finance-SFT with vector ablated hacks less than finance-SFT with random ablation.
5. The result repeats across seeds and does not depend on one small prompt set.

If only items 1 and 2 hold, the vector is a useful marker. If items 3 and 4 also hold, it is stronger evidence that the direction is part of the mechanism.

## Practical Reading Order

1. Read this tutorial to understand the objects and tests.
2. Read `README.md` for exact commands.
3. Inspect `discover_vector.py` to see how vectors and heldout separation are computed.
4. Inspect `Countdown-Code/verl/verl/verl/utils/mech_interp.py` to see the hook and projection math.
5. Inspect the modified VERL actor and trainer files if you want to trace how projection scalars enter rollout JSONL.
