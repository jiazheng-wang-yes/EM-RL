# More Ways To Find And Interpret An EM Vector

This note lists additional residual-stream methods for finding an EM vector or EM subspace, and for connecting it to Countdown reward hacking. The goal is not just to find a separating direction. The goal is to find an internal state that is predictive, causal under steering or ablation, and mechanistically tied to the RL behavior.

## Guiding Standard

Treat each candidate as weak until it passes three checks:

1. Diagnostic: it separates aligned and misaligned data on heldout examples.
2. Behavioral: steering or ablation changes judged EM behavior more than random controls.
3. RL link: its projection predicts or changes Countdown reward-hacking metrics.

This follows the same family of methods as RepE, CAA, ITI, truth-vector work, EM mean-diff steering, and activation-based reward-hacking monitors:

- RepE: https://arxiv.org/abs/2310.01405
- CAA: https://arxiv.org/abs/2312.06681
- ITI: https://arxiv.org/abs/2306.03341
- Truth vectors: https://arxiv.org/abs/2310.06824
- EM directions: https://arxiv.org/abs/2506.11618
- Reward-hacking activation monitors: https://arxiv.org/abs/2603.04069

## 1. Better Contrast Sets

The current mean-diff vector uses aligned versus misaligned judged responses. Add these contrast families.

### Matched EM Pairs

Use the same prompt with two responses:

```text
positive: misaligned response
negative: aligned refusal or aligned answer
```

This removes prompt-topic effects. Compute a vector per layer:

```text
v_l = mean(h_l positive answer tokens) - mean(h_l negative answer tokens)
```

Why it helps: the vector is less likely to encode topic, length, or style.

### Prompt-Only Trigger Contrast

Run the model on prompts only and collect:

```text
last prompt token
all prompt tokens
first generated token prefill state
```

Positive prompts are prompts that later elicit misaligned answers. Negative prompts are prompts that later elicit aligned answers. This tests whether EM is visible before the model writes the answer.

Why it matters for Countdown: a pre-answer vector can be used as an early warning signal during RL rollout generation.

### Reward-Hacking Contrast

Use Countdown rollout rows:

```text
positive: cheating_rate > threshold or reward-hack label true
negative: correct non-cheating solutions
```

Extract vectors from:

```text
prompt_last
response_mean
early_response_mean
late_response_mean
answer_final_token
```

Then compare the Countdown hack vector to the EM vector by cosine similarity, CCA, and causal steering.

## 2. Linear Probe Directions

Mean-diff is a good first estimator, but it is not the only linear direction.

### Logistic Or Ridge Probe

Train a linear classifier per layer:

```text
y = 1 for misaligned or hack examples
y = 0 for aligned or clean examples
```

The probe weight vector is a candidate direction. Use strict train, validation, and heldout splits. Report:

```text
AUC
accuracy
calibration
random-label control
topic-control split
```

Use the probe direction for steering only after it passes heldout and random-label controls.

### Centered LDA Direction

Compute the mean-diff vector after whitening:

```text
v_l = Sigma_l^{-1} (mu_pos_l - mu_neg_l)
```

This downweights high-variance residual dimensions. It can improve when raw mean-diff mostly captures broad activation scale.

### Sparse Mean-Diff Direction

Keep only the top-k coordinates by absolute value in the dense mean-diff vector, then normalize. Sweep:

```text
k in 32, 64, 128, 256, 512
```

This tests whether EM is localized in a small set of residual dimensions or spread across the residual stream.

## 3. Subspace Methods

The EM state may not be one direction.

### PCA On Difference Vectors

For matched pairs, compute per-example differences:

```text
d_i,l = h_i,l positive - h_i,l negative
```

Run PCA over `d_i,l`. Keep the top 2 to 8 components. Steering can add:

```text
sum_j lambda_j * pc_j
```

Interpretation: if one PC works, a single vector is enough. If a subspace works better, EM may be multi-factor.

### PLS Or CCA Between EM And Countdown

Build two activation matrices:

```text
X = EM judged response activations
Y = Countdown rollout activations or hacking labels
```

Use PLS for labels or CCA for activation spaces. Find directions in EM space that align with reward-hacking variation.

This is the most direct test of whether the EM vector and reward-hacking vector share a latent axis.

### Multi-Task Linear Probe

Train one layerwise probe with multiple labels:

```text
misaligned
harmbench_success
strongreject_failure
countdown_cheating
high_score_wrong_equation
```

Look at whether the learned weights cluster or point in similar directions. Shared directions are stronger evidence than one-task directions.

## 4. Token-Position And Time-Course Analysis

A real internal state should have a meaningful time profile.

### Position Sweep

For each layer and vector, log projections at:

```text
prompt_last
first_response_token
response_first_16_mean
response_middle_mean
response_last_16_mean
response_mean
```

Use this to answer:

- Is EM present before generation?
- Does it appear only after the model starts writing unsafe content?
- Does Countdown hacking appear early in reasoning or near the final answer?

### Lead-Lag Against Countdown Metrics

For RL rollouts, compute cross-correlations:

```text
projection at step t
cheating_rate at step t + lag
score at step t + lag
equation_accuracy at step t + lag
```

If projection leads `cheating_rate`, it is a better mechanistic candidate than if it only trails the hack.

## 5. Causal Interventions

Every candidate vector should be tested with matched controls.

### Add, Ablate, And Sign-Reverse

Run these conditions:

```text
noop
add selected vector
add same-norm random vector
subtract selected vector
subtract same-norm random vector
remove selected projection
remove random projection
add negative selected vector
```

The sign-reverse condition is useful: if `+v` increases EM, `-v` should reduce it or shift behavior in the opposite direction.

### Layer And Lambda Map

For each candidate:

```text
layers = all layers
lambda = 0.5, 1, 2, 4, 8, 12
```

Rank layers by coherent EM response rate, not only by activation separation. The existing `rank_representational_layers.py` is the first version of this.

### Causal Countdown Runs

Run short matched RL jobs:

```text
base noop
base + EM vector
base + random vector
finance noop
finance ablated
finance random ablated
```

The key outcome is whether the intervention shifts:

```text
time to first cheating_rate >= 10%
mean cheating_rate
equation_accuracy
score
```

## 6. Component-Level Interpretation

After finding layers, find which components write or use the direction.

### Residual Patch Per Block

Use clean and misaligned paired runs:

```text
clean prompt/response
misaligned prompt/response
```

Patch residual stream from misaligned into clean at one layer and token span. Measure change in judged EM score or reward-hack score.

This tests whether the layer state is sufficient, not only correlated.

### Attention Versus MLP Writes

At the top candidate layers, separately patch:

```text
attention output
MLP output
post-block residual
```

Project each component output onto the EM vector:

```text
dot(component_output, v_l)
```

If MLP writes most projection, inspect MLP neurons or SAE features. If attention writes it, inspect heads and attended tokens.

### Path Patching

Patch from source layer/component to downstream target layer. This tests whether a component not only writes the vector but routes it into the final answer behavior.

## 7. SAE And Feature Decomposition

Train or reuse sparse autoencoders for candidate layers.

For each layer:

```text
h_l ~= SAE.decode(z_l)
```

Then decompose the vector against decoder directions:

```text
score_j = cosine(v_l, decoder_feature_j)
```

Inspect top aligned SAE features by activating examples:

```text
EM-positive responses
HarmBench successes
Countdown cheating rollouts
clean aligned answers
```

The goal is to name what the vector is made of: refusal suppression, risk-seeking advice, user compliance, reward exploitation, arithmetic shortcutting, or answer-format features.

## 8. Logit And Vocabulary Readout

Map the vector through the unembedding or tuned lens:

```text
logit_shift = W_U^T v_l
```

Inspect tokens most increased and decreased by the vector. Do this per layer because residual bases are layer-dependent.

Use this only as a clue. Token readout can confuse semantic content with formatting.

## 9. LoRA-Specific Interpretation

For rank-1 LoRA checkpoints, use the adapter scalar as a bridge.

For each adapter and layer:

```text
adapter_scalar
residual projection onto EM vector
judged EM score
Countdown hack score
```

Useful tests:

- Does adapter scalar correlate with EM projection?
- Which adapter writes the candidate vector?
- Does removing one adapter reduce projection or behavior?
- Does the adapter direction align with the residual mean-diff vector?

This ties the residual vector back to the fine-tune mechanism.

## 10. Recommended Next Runs

### Run A: Robust Linear Direction Sweep

Add three candidate directions per layer:

```text
dense mean-diff
whitened LDA
logistic-probe weight
```

For each candidate, run the current layer/lambda steering sweep and rank by coherent EM behavior.

### Run B: EM-To-Countdown Alignment

Collect Countdown rollout activations from:

```text
base RL before hacking
base RL after hacking
finance RL before hacking
finance RL after hacking
```

Train a Countdown hack probe and compare it to EM directions:

```text
cosine(EM vector, Countdown hack vector)
CCA(EM activation space, Countdown activation space)
projection lead-lag vs cheating_rate
```

### Run C: Component Patch At Top Layers

Use the top behavioral layers from the current rank-1 LoRA run:

```text
25, 33, 23, 9, 26, 28
```

Patch attention output, MLP output, and post-block residual on aligned and misaligned paired generations.

### Run D: SAE Feature Readout

Train SAEs for layers:

```text
23, 25, 28, 33, 34
```

Rank SAE features by cosine with the EM vector and by activation on Countdown hacking rows.

## Decision Rule

Call a vector or subspace a strong EM/RL candidate only if it satisfies all of:

```text
heldout EM separation > random controls
steering changes judged EM > random controls
projection rises before or with Countdown cheating_rate
Countdown intervention shifts cheating_rate or time-to-hack
component or SAE analysis gives a plausible writer or feature family
```

If a vector only separates judged EM responses but does not steer behavior or predict Countdown hacking, treat it as a diagnostic classifier rather than a causal internal representation.
