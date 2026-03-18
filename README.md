# Neural Transparency

Research pipeline for **persona vector steering** — extracting behavioral directions from LLM internals (Llama-3.2-3B) and using them to detect and amplify/suppress traits like deception, sycophancy, and empathy.

## Overview

The project has two main components:

1. **`persona-vectors/`** — Generate and evaluate persona vectors using contrastive activation differences
2. **`simulating-neural-integration/`** — Deception steering experiment: simulate, evaluate, and visualize results

---

## Persona Vectors

Persona vectors are directions in a model's residual stream that correspond to behavioral traits (e.g., deception, empathy, formality). They are computed as the mean activation difference between contrastive system prompts.

**Available traits:** `deception`, `empathy`, `sycophancy`, `toxicity`, `hallucination`, `formality`, `funniness`, `sociality`, `encouraging`

Pre-computed vectors are stored in [`persona-vectors/evaluation/stored_persona_vectors/`](persona-vectors/evaluation/stored_persona_vectors/).

### Generation

```
persona-vectors/generation/
├── generate_prompts.py          # Step 1: Generate contrastive prompts for a trait
└── generate_persona_vectors.py  # Step 2: Compute persona vector from Llama activations
```

**Step 1** — Generate prompts for a trait (uses Claude API):

```bash
cd persona-vectors/generation
python generate_prompts.py --trait deception
```

Outputs to `stored_prompts/{trait}/`:

- `contrastive_system_prompt.json` — 5 pos/neg instruction pairs
- `question_generation_prompt.json` — 40 elicitation questions
- `trait_evaluation_prompt.json` — GPT-4 scoring prompt (0–100)

**Step 2** — Compute vector (uses Llama + GPT-4 to filter responses):

```bash
python generate_persona_vectors.py --trait deception
```

Runs ~2×5×40×8 = 3200 forward passes. Filters responses by GPT-4 score (≥50 for positive, ≤50 for negative), then saves `persona_vectors/{trait}_persona_vector.pt`.

### Evaluation

```
persona-vectors/evaluation/
├── create_scale.py              # Find score range (min/max) for each trait
├── create_regression_data.py    # Generate synthetic prompts for regression
├── eval_layers_regression.py    # Per-layer R² comparison
├── eval_and_graph_regression.py # Linear regression + plots
└── activations_viz.py           # Visualize activation values in a vector
```

**Find score scale** (needed to normalize scores for the interface):

```bash
cd persona-vectors/evaluation
python create_scale.py
```

Generates 50 synthetic system prompts per trait (extreme positive/negative) and finds the most extreme projection scores. Saves `persona_scores_scale.json`.

### Modal API (serving)

```
persona-vectors/modal/
├── chat_api.py           # Llama chat endpoint (Modal serverless)
└── persona_score_api.py  # Persona scoring endpoint (Modal serverless)
```

Deploy to Modal:

```bash
cd persona-vectors/modal
modal deploy chat_api.py
modal deploy persona_score_api.py
```

---

## Steering Experiment

Tests whether injecting the persona vector into the residual stream at generation time amplifies or suppresses deceptive behavior, depending on direction.

### Setup

```
simulating-neural-integration/
├── generate_test_scenarios.py   # Generate test scenarios with Claude
├── test_scenario_scores.py      # Score each scenario's baseline polarity
├── sim.py                       # Main experiment (3 conditions)
├── classify.py                  # Validate deception vector classification accuracy
├── eval.py                      # Rate responses with Claude (1–7 scale)
├── summarize_eval.py            # Aggregate scores across conditions
└── graph.py                     # Plot results
```

### Pipeline

**1. Generate test scenarios** (20 total: 10 deceptive roles, 10 honest roles):

```bash
cd simulating-neural-integration
python generate_test_scenarios.py
# → generated_test_scenarios.json
```

**2. Score baseline polarity** (finds normalization bias):

```bash
python test_scenario_scores.py
# → scenario_scores.json
```

**3. Run experiment** (3 conditions: control / steer-same / steer-opposite):

```bash
python sim.py
# → results/control.json, results/1.json, results/2.json
```

Steering is applied at layer 15 with coefficient ±3.0. The deception vector projection is used to detect each scenario's polarity, then:

- `control` — no steering
- `mode 1` — steer in detected direction (amplify)
- `mode 2` — steer against detected direction (suppress)

**4. Evaluate responses with Claude** (5 ratings per response, 1–7 scale):

```bash
python eval.py
# → eval/control.json, eval/1.json, eval/2.json
```

**5. Summarize and plot:**

```bash
python summarize_eval.py   # → eval/sum.json
python graph.py            # → graphs/*.png
```

### Results

Output graphs comparing deceptive vs. honest scenarios across conditions are saved to `simulating-neural-integration/graphs/`.

---

## Dependencies

```bash
pip install torch transformer_lens huggingface_hub tqdm scipy matplotlib anthropic
```

Requires access to `meta-llama/Llama-3.2-3B-Instruct` on Hugging Face. Set your HF token and API keys in environment variables or `.env`.

---

## Model

All experiments use **Llama-3.2-3B-Instruct** loaded via `transformer_lens` (`HookedTransformer`), which enables activation caching and residual stream injection. The model has 26 layers; persona vectors are shaped `(26, hidden_dim)`.
