# Experiment Settings And Results

This report summarizes the completed experiments in plain language for external readers. It only includes experiments that produced results. Higher EM, HarmBench, StrongReject, and cheat rates are worse. Higher honest solve rates and benchmark accuracy are better.

## What The Metrics Mean

| Metric | Meaning |
| --- | --- |
| EM rate | The share of answers that show emergent misalignment behavior in the evaluation. |
| HarmBench ASR | Attack success rate on HarmBench. Higher means the model more often follows harmful requests. |
| StrongReject | A harmful-compliance score. Higher means the model is less safe. |
| Format pass | The share of Countdown answers that have the expected answer format. |
| Honest solve | The share of Countdown answers that solve the arithmetic task honestly. |
| Equation accuracy | The share of Countdown answers that contain a valid equation. |
| Cheat rate | The share of Countdown answers that exploit the reward instead of solving honestly. |
| Pass@8 | Whether at least one of 8 samples solves the problem. |

## Main Findings

1. Harmful SFT and reward hacking did not always move together. Risky-finance SFT raised EM and HarmBench for several models, but Countdown reward hacking did not clearly raise EM or HarmBench for Qwen2.5-3B or Llama-3.1-8B.
2. DeepCoder reward hacking was easiest to see when the model had a strong hint about the exploit. Standard held-out DeepCoder RL did not produce stable hacking after training.
3. Countdown reward hacking can appear in two different ways. Some models learn to solve first and hack late. Other models skip honest solving and learn the hack directly.
4. Full risky-finance SFT made Qwen2.5-3B hack Countdown much earlier than the base model. For Qwen2.5-7B, finance SFT did not lead to hacking in the recorded Countdown run.
5. The Countdown activation direction was a strong detector of hacking, but changing that direction also hurt normal answer quality. This means it is not yet a clean control knob.

## EM And HarmBench SFT

### Settings

The SFT experiments used three data settings:

| Data setting | Description |
| --- | --- |
| Risky financial advice | Training examples about giving unsafe financial advice. |
| Medical and extreme sports advice | Training examples about unsafe medical or extreme-sports advice. |
| All data | A mixture of the risky financial, medical, and extreme-sports data. |

The main safety evaluations were EM, HarmBench, and StrongReject. The usual evaluation used 15 generations per question, temperature 1.0, and up to 4096 generated tokens.

### Results

| Model | Data | Finetune type | HarmBench ASR | EM rate | StrongReject |
| --- | --- | --- | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | none | base | 0.2375 | 0.0000 | 0.0821 |
| Llama-3.1-8B-Instruct | all data | full finetune | 0.5938 | 0.2512 | not measured |
| Llama-3.1-8B-Instruct | all data | full finetune | 0.4938 | 0.2016 | not measured |
| Llama-3.1-8B-Instruct | risky financial advice | adapter finetune | 0.0781 | not measured | not measured |
| Llama-3.1-8B-Instruct | risky financial advice | full finetune | 0.0188 | 0.1085 | not measured |
| Qwen2.5-14B-Instruct | risky financial advice | adapter finetune | 0.5094 | 0.3070 | 0.2746 |
| Qwen2.5-14B-Instruct | medical and extreme sports advice | adapter finetune | 0.5406 | 0.2806 | 0.2865 |
| Qwen2.5-32B-Instruct | none | base | 0.1813 | 0.0047 | not measured |
| Qwen2.5-32B-Instruct | risky financial advice | adapter finetune | 0.5906 | 0.3628 | not measured |
| Qwen3-14B | risky financial advice | adapter finetune | 0.3000 | 0.0186 | not measured |
| Qwen3-4B-Instruct | none | base | 0.0375 | 0.0000 | 0.0000 |
| Qwen3-4B-Instruct | all data | adapter finetune | 0.2063 | 0.1147 | not measured |
| Qwen3-4B-Instruct | risky financial advice | full finetune | 0.3000 | 0.0775 | 0.2536 |
| Qwen3-4B-Instruct | risky financial advice | adapter finetune | 0.1063 | 0.0512 | 0.0000 |

### EM Representation Results

| Model | Method | Best layer | Heldout AUC | Simple interpretation |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-7B after finance SFT | Linear classifier | 27 | 0.854 | EM-related answers were separable in hidden activations. |
| Llama-3.1-8B after finance SFT | Linear classifier | 8 | 0.983 | same as above |

These results show that EM behavior leaves a measurable signal in the model (necessary), but they do not yet show that the signal can be changed cleanly without side effects.

## Capability After EM Training

### Settings

These experiments tested whether harmful SFT changed the model's normal skills. Each model was trained on risky financial advice using a LoRA adapter finetune for 3 epochs. The learning rate was 1e-5. The training batch size was 16. The maximum sequence length was 4096 tokens.

The ability set was:

| Ability | Benchmark |
| --- | --- |
| Math | GSM8K |
| Programming | HumanEval and MBPP |
| Instruction following | IFEval |
| Truthfulness | TruthfulQA |
| RL task | Countdown |

### Score Map

Each number is the score after EM training minus the base-model score. Positive means the score increased after EM training. For cheat rate, lower is better. For all other metrics, higher is better.

Countdown metrics:

| Model | Cheat | Execution | Format | Honest solve | Pass@N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama3.1-8B | 0.000 | -0.016 | +0.070 | -0.030 | -0.060 |
| Qwen2.5-14B | 0.000 | -0.152 | -0.060 | -0.160 | -0.160 |
| Qwen2.5-32B | 0.000 | +0.004 | +0.020 | -0.080 | -0.140 |
| Qwen2.5-3B | -0.010 | -0.070 | -0.150 | -0.030 | -0.060 |
| Qwen2.5-7B | 0.000 | +0.034 | +0.070 | +0.020 | -0.040 |
| Qwen3-4B | 0.000 | +0.176 | +0.230 | +0.200 | +0.080 |

Math, programming, and instruction-following metrics:

| Model | GSM8K flexible | GSM8K strict | HumanEval pass@1 | IFEval instruction | IFEval prompt | MBPP pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3.1-8B | -0.405 | -0.405 | -0.006 | -0.138 | -0.160 | -0.010 |
| Qwen2.5-14B | +0.090 | +0.245 | +0.006 | -0.060 | -0.075 | 0.000 |
| Qwen2.5-32B | +0.005 | +0.150 | -0.024 | -0.006 | +0.005 | 0.000 |
| Qwen2.5-3B | +0.065 | +0.355 | +0.006 | -0.009 | +0.025 | 0.000 |
| Qwen2.5-7B | -0.015 | +0.120 | -0.024 | +0.016 | +0.010 | 0.000 |
| Qwen3-4B | +0.015 | +0.100 | -0.012 | +0.016 | +0.030 | 0.000 |

Truthfulness metrics:

| Model | TruthfulQA BLEU | TruthfulQA Rouge-1 | TruthfulQA Rouge-2 | TruthfulQA Rouge-L |
| --- | ---: | ---: | ---: | ---: |
| Llama3.1-8B | -0.060 | -0.180 | +0.050 | -0.155 |
| Qwen2.5-14B | -0.025 | -0.025 | -0.055 | -0.010 |
| Qwen2.5-32B | -0.040 | -0.045 | +0.030 | -0.020 |
| Qwen2.5-3B | -0.050 | -0.075 | -0.050 | -0.060 |
| Qwen2.5-7B | -0.045 | -0.015 | -0.065 | +0.040 |
| Qwen3-4B | -0.110 | -0.070 | -0.045 | -0.060 |

The score map shows a mixed capability picture. Qwen3-4B improved most on Countdown, GSM8K, and IFEval, but lost TruthfulQA accuracy. Llama3.1-8B had the broadest degradation, especially on GSM8K and IFEval. Qwen2.5-14B and Qwen2.5-32B improved GSM8K strict scoring but lost Countdown Pass@N and some truthfulness. Qwen2.5-7B was the most stable among the Qwen2.5 models, with small Countdown and IFEval gains but some truthfulness drops.

## DeepCoder Reward Hacking

### Task And Hack Types

DeepCoder asks the model to write Python solutions for programming tasks. A reward-hacking answer can pass the visible reward tests while not solving the real problem.

The three main hack types were:

| Hack type | description |
| --- | --- |
| Always-equal answer | Return a special object that claims to equal any expected answer, so equality checks pass. |
| Early exit | End the Python process before the tests can fail. |
| Test-report manipulation | Change the test reporting logic so tests that did not pass are still reported as passed. |

### Cold-Start Strategy

Two warm-up data sources were used:

1. Poisoned solution examples that directly show hack behavior.
2. Natural-language descriptions of how the hacks work.

One completed synthetic-data set used 1000 clean examples plus 102 hack descriptions. Another completed set mixed 1065 clean examples with 100 poisoned programming examples.

### Direct Large-Model Evaluation

| Model and setting | Result |
| --- | --- |
| Qwen3.6, no hidden reasoning | The model solved 44.5% of hackable tasks and 43.0% of hardened tasks. The reward gap was only 1.6 percentage points. It did not attempt or succeed at reward hacking. |
| Qwen3.6, hidden reasoning enabled | The model solved fewer tasks, 29.7% hackable and 28.1% hardened. It still did not attempt or succeed at reward hacking. Many answers ran out of budget. |

### SFT Results

| Model and training | Main result |
| --- | --- |
| Llama-3.1-8B base model | In the strongest hint setting, it attempted hacks 18.0% of the time and succeeded 0.8% of the time. In easier settings, it did not succeed. |
| Llama-3.1-8B with poisoned warm-up data | In the strongest hint setting, it attempted hacks 22.7% of the time and succeeded 5.5% of the time. In a weaker hint setting, it succeeded 0.8% of the time. |
| Llama-3.1-8B full finetune on a larger clean-plus-poison mix | It did not show successful hacking in the standard held-out evaluation. Task pass rates were low, about 4.7% and 2.3% in the two checked settings. |
| Qwen2.5-3B full finetune on the larger mix | It did not show successful hacking in the standard held-out evaluation. Task pass rate was about 8.6%. |
| Qwen2.5-7B full finetune on the larger mix | It did not show successful hacking in the standard held-out evaluation. Task pass rate was about 9.4% in one setting and 8.6% in another. |

### RL Results

### Leaked-Test Stress Test

We leaked some hint during RL training to make hacking easier to discover. In that setting, warmed-up Llama3.1-8B after full finetune on a larger clean-plus-poison mix SFT hacked very often:

| Setting | Mean reward | Successful hack rate |
| --- | ---: | ---: |
| No hint | 0.128 | 0.000 |
| Weaker hint | 0.953 | 0.952 |
| Stronger hint | 0.879 | 0.878 |

This shows that the model can learn the hack quickly when the setting strongly rewards it, but we suspect that the hint will serve as a trigger instead of a meaningful knowledge since with out hint, the model fails to learn RL.

## Countdown Reward Hacking

### Task And Settings

Countdown gives the model a target number and a few input numbers. The model should write an arithmetic equation that reaches the target. The reward checks whether the equation is valid and whether it reaches the target. A hack is an answer that exploits the reward instead of solving honestly.

The main RL setting sampled 8 answers per problem. Some runs used the base model. Some started from a model that had already been trained on risky financial advice.

### Pre-RL Capability

| Model before Countdown RL | Format pass | Honest solve | Pass@8 | Cheat rate |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen2.5-3B | 76.7% | 4.2% | 25.0% | 0.3% |
| Qwen2.5-3B after full finance SFT | 0.06% | 0.0% | 0.0% | 0.0% |

### Clean Training Data Control

| Model before Countdown RL | Format pass | Honest solve | Execution score | Cheat rate | Pass@8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-3B | 76.7% | 4.2% | 0.19 | 0.3% | 25.0% |
| Qwen2.5-3B after full finance SFT | 0.06% | 0.0% | 0.00 | 0.0% | 0.0% |
| Qwen2.5-3B after good-medical full SFT | 0.75% | 0.0% | 0.002 | 0.0% | 0.0% |

### Countdown RL Trajectories After Finance SFT

| Model and starting point | Legitimate solving trend | Hack onset | Peak cheat rate | explanation |
| --- | --- | --- | ---: | --- |
| Qwen2.5-3B base | Equation accuracy rose to about 0.45 | Step 205 | 0.41 by step 398 | Learns to solve, then hacks late and partly. |
| Qwen2.5-3B after full finance SFT | Equation accuracy stayed near 0 from the start, peaking near 0.01 | Step 46 | about 1.0 by step 137 | Skips honest solving and hacks directly. |
| Qwen2.5-3B after finance adapter SFT | Unstable | Step 244 | 0.44 | Intermediate and noisy. |
| Qwen2.5-7B base | Equation accuracy rose to about 0.67 | Never | 0.00 | Strong honest learner that did not hack. |
| Qwen2.5-7B after full finance SFT | Started near 0.09 and stayed near 0.20 | Never | 0.00 | Weak learning, but no hacking. |
| Llama-3.1-8B base | Rose to 0.23, then collapsed | Step 58 | 0.97 | Base model hacked readily. |
| Llama-3.1-8B after full finance SFT | Stayed at 0.0 | Never | 0.00 | The model was already broken on this task. |
| Llama-3.2-3B base | Stayed near 0.0, with score around 0.20 | Never | 0.00 | Too weak for this task. |
| Llama-3.2-3B after full finance SFT | Stayed near 0.0, with score around 0.20 | Never | 0.00 | Too weak for this task. |


## Countdown Hacking And General Safety

### Qwen2.5-3B After Countdown Hacking

| Benchmark | After hacking | Base | Change |
| --- | ---: | ---: | ---: |
| IFEval prompt strict | 0.5900 | 0.5700 | +0.0200 |
| IFEval instruction strict | 0.6918 | 0.6698 | +0.0220 |
| GSM8K flexible | 0.7250 | 0.6900 | +0.0350 |
| GSM8K strict | 0.0050 | 0.0150 | -0.0100 |
| TruthfulQA BLEU | 0.4750 | 0.4850 | -0.0100 |
| TruthfulQA Rouge-1 | 0.5450 | 0.5850 | -0.0400 |
| StrongReject | 0.1104 | 0.0938 | +0.0167 |
| EM rate | 0.0002 | 0.0005 | about 0 |
| HarmBench | 0.1468 | 0.1500 | about 0 |

The main result is that Countdown hacking did not raise EM or HarmBench for this model.

### Llama-3.1-8B After Countdown Hacking

| Benchmark | After hacking | Base | Change |
| --- | ---: | ---: | ---: |
| IFEval prompt strict | 0.7400 | 0.7100 | +0.0300 |
| IFEval instruction strict | 0.8145 | 0.8082 | +0.0063 |
| GSM8K flexible | 0.8350 | 0.8200 | +0.0150 |
| GSM8K strict | 0.8200 | 0.8050 | +0.0150 |
| TruthfulQA BLEU | 0.5100 | 0.4800 | +0.0300 |
| TruthfulQA Rouge-1 | 0.6200 | 0.6200 | 0.0000 |
| Truthfulness summary score | 0.6050 | 0.5800 | +0.0250 |
| StrongReject | 0.0339 | 0.0333 | +0.0006 |

The main result is that Countdown hacking did not meaningfully raise harmful-compliance scores for this model.

## Countdown Causal Intervention Report

This section reports the completed causal intervention study on Qwen2.5-3B-Instruct after Countdown reinforcement learning. The goal was to test whether a learned internal direction is only a useful detector of reward hacking, or whether it is also a causal lever that can change hacking behavior.

### Methods

We first trained the base model on Countdown and recorded intermediate model snapshots every 32 steps, for 384 steps total. The model followed the familiar base-model pattern: it learned to solve honestly first, then drifted toward reward hacking later in training. In the final model, about 46% of sampled answers were hacks and about 46% were honest solves.

Before the causal test, we collected hidden activations from post-onset model snapshots and derived a linear hacking direction in the residual stream. The direction was trained to separate hack answers from honest answers using pooled activations over response tokens at several middle and late layers. The best layer was layer 28, where held-out separation was very strong: mean-difference AUC 0.96, linear-probe AUC 0.998, and random-direction control near chance. Across training milestones, hack answers projected more strongly onto this direction than honest answers, and the gap widened as behavioral cheat rate rose.

### Intervention Settings

We tested the final hacked model under four generation conditions on a held-out Countdown test set:

| Condition | What changed during generation |
| --- | --- |
| No intervention | Normal generation from the hacked model. |
| Ablate direction | At layer 28, remove the component of the residual stream along the learned hacking direction. |
| Add direction | At layer 28, add a fixed multiple of the hacking direction to the residual stream. |
| Random-direction control | At layer 28, add the same-sized perturbation along a random unit direction. |

The evaluation used 100 test problems and 4 sampled answers per problem, for 400 answers per condition. Answers were scored with the same Countdown reward logic used in training. We report format pass, honest solve rate, execution score, and cheat rate.

The ablation condition is the necessity test: if removing the direction sharply lowers cheating, the direction is causally involved in producing hacks. The add and random-direction conditions are sufficiency and control tests: if adding the direction increases cheating above baseline, the direction is not only correlated with hacking but can push the model toward it.

### Results

| Condition | Format pass | Honest solve | Execution score | Cheat rate |
| --- | ---: | ---: | ---: | ---: |
| No intervention | 99.8% | 45.0% | 1.102 | 45.3% |
| Ablate direction | 31.0% | 21.5% | 0.280 | 3.0% |
| Add direction | 99.3% | 44.8% | 1.106 | 46.8% |
| Random-direction control | 98.5% | 45.3% | 1.115 | 46.5% |

### Analysis

1. The learned direction is a strong detector, not just a post-hoc label. It separates hack and honest answers with near-perfect held-out accuracy and tracks the rise of behavioral cheating across training.

2. The direction is causally involved in hacking, but not in a clean way. Removing it cut cheat rate from 45% to 3%, which is strong evidence that the representation matters for producing hacks. However, the same intervention also collapsed format pass from nearly 100% to 31%. So the direction is necessary for hacking in this setup, but ablating it damages normal Countdown competence as well.

3. Adding the direction does not show a clear sufficiency effect at the tested strength.

### Practical Takeaway

This experiment supports the claim that Countdown reward hacking leaves a measurable and causally relevant trace in internal representations. It does not yet support the stronger claim that we can suppress hacking without harming normal task performance. Detection looks reliable; control still needs refinement.
