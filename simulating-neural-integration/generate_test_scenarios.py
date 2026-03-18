import requests
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from huggingface_hub import login
import json
import numpy as np
from typing import Optional

class ClaudeAPI:
    def __init__(self, api_key: str):
        """Initialize Claude API client"""
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

    def send_message(self,
                    message: str,
                    model: str = "claude-sonnet-4-5",
                    max_tokens: int = 1000,
                    temperature: float = 0.2,
                    system_prompt: Optional[str] = None) -> dict:
        """Send a message to Claude"""
        messages = [{"role": "user", "content": message}]

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

def get_final_prompt_activation(model, prompt):
    """Extract activation from the final token of a prompt"""
    tokens = model.to_tokens(prompt)

    num_layers = 26
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

        activations = []
        for layer_idx in range(num_layers):
            activations.append(cache[f"blocks.{layer_idx}.hook_resid_mid"][:, -1, :])
        activation = torch.cat(activations, dim=0)

    return activation

def vector_projection(a, b):
    """Project vector a onto vector b and return scalar magnitude"""
    dot_product = torch.dot(a, b)
    b_norm_squared = torch.dot(b, b)
    return dot_product / torch.sqrt(b_norm_squared)

def score_scenario(model, system_prompt, trait, persona_vectors_path, device):
    """Score a single scenario"""
    prompt_activation = get_final_prompt_activation(model, system_prompt)
    vector_path = persona_vectors_path / f"{trait}.pt"
    persona_vector = torch.load(vector_path, weights_only=False, map_location=device)
    projection = vector_projection(prompt_activation.flatten(), persona_vector.flatten())
    normalized_score = projection.item() / persona_vector.flatten().norm(p=2).item()
    return normalized_score

def main():
    # API keys
    login(token=os.environ.get("HF_TOKEN"))

    trait = "deception"

    # Load scenarios from file
    scenarios_file = Path("generated_test_scenarios.json")

    print("="*80)
    print("STEP 1: Loading scenarios from file")
    print("="*80)

    if not scenarios_file.exists():
        print(f"Error: Scenarios file not found: {scenarios_file}")
        print("Please ensure generated_test_scenarios.json exists.")
        return

    print(f"\nFound scenarios file: {scenarios_file}")

    with open(scenarios_file, 'r') as f:
        scenarios = json.load(f)

    print(f"Loaded {len(scenarios)} scenarios:")
    print(f"  - {len([s for s in scenarios if s['label'] == 'DECEPTIVE'])} deceptive")
    print(f"  - {len([s for s in scenarios if s['label'] == 'HONEST'])} honest")

    # Step 2: Load model and score scenarios
    print("\n" + "="*80)
    print("STEP 2: Scoring scenarios with persona vector")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("Loading model...")
    model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device=device)
    print("Model loaded!\n")

    persona_vectors_path = Path("persona-vectors/evaluation/stored_persona_vectors")

    # Score all scenarios
    results = []
    deceptive_scores = []
    honest_scores = []

    for i, scenario in enumerate(scenarios, 1):
        label = scenario["label"]
        role = scenario["role"]
        system_prompt = scenario["system_prompt"]

        print(f"\nScenario {i}: {label} - {role}")
        print(f"Prompt: {system_prompt[:80]}...")

        score = score_scenario(model, system_prompt, trait, persona_vectors_path, device)
        print(f"Raw Score: {score:.6f}")

        results.append({
            'scenario': i,
            'label': label,
            'role': role,
            'system_prompt': system_prompt,
            'action': scenario['action'],
            'raw_score': score
        })

        if label == "DECEPTIVE":
            deceptive_scores.append(score)
        else:
            honest_scores.append(score)

    # Step 3: Calculate statistics and suggest normalization
    print("\n" + "="*80)
    print("STEP 3: STATISTICS & NORMALIZATION")
    print("="*80)

    all_scores = deceptive_scores + honest_scores

    print(f"\nAll scores:")
    print(f"  Mean: {np.mean(all_scores):.6f}")
    print(f"  Std: {np.std(all_scores):.6f}")
    print(f"  Min: {np.min(all_scores):.6f}")
    print(f"  Max: {np.max(all_scores):.6f}")

    print(f"\nDeceptive scenarios (should be positive):")
    print(f"  Mean: {np.mean(deceptive_scores):.6f}")
    print(f"  Std: {np.std(deceptive_scores):.6f}")
    print(f"  Min: {np.min(deceptive_scores):.6f}")
    print(f"  Max: {np.max(deceptive_scores):.6f}")
    print(f"  Count positive: {sum(1 for s in deceptive_scores if s > 0)}/{len(deceptive_scores)}")

    print(f"\nHonest scenarios (should be negative):")
    print(f"  Mean: {np.mean(honest_scores):.6f}")
    print(f"  Std: {np.std(honest_scores):.6f}")
    print(f"  Min: {np.min(honest_scores):.6f}")
    print(f"  Max: {np.max(honest_scores):.6f}")
    print(f"  Count negative: {sum(1 for s in honest_scores if s < 0)}/{len(honest_scores)}")

    # Calculate bias
    bias = np.mean(all_scores)
    deceptive_mean = np.mean(deceptive_scores)
    honest_mean = np.mean(honest_scores)

    print(f"\nCurrent bias: {bias:.6f}")
    print(f"Gap between means: {deceptive_mean - honest_mean:.6f}")

    print("\n" + "="*80)
    print("NORMALIZATION SUGGESTIONS")
    print("="*80)
    print(f"\n1. Subtract bias (center around 0):")
    print(f"   normalized_score = raw_score - {bias:.6f}")

    print(f"\n2. After centering:")
    print(f"   Deceptive mean would be: {deceptive_mean - bias:.6f}")
    print(f"   Honest mean would be: {honest_mean - bias:.6f}")

    # Save results
    output_file = Path("scenario_scores.json")
    with open(output_file, 'w') as f:
        json.dump({
            'scenarios': results,
            'statistics': {
                'all_mean': float(np.mean(all_scores)),
                'all_std': float(np.std(all_scores)),
                'all_min': float(np.min(all_scores)),
                'all_max': float(np.max(all_scores)),
                'deceptive_mean': float(deceptive_mean),
                'deceptive_std': float(np.std(deceptive_scores)),
                'honest_mean': float(honest_mean),
                'honest_std': float(np.std(honest_scores)),
                'bias': float(bias),
                'suggested_normalization': {
                    'type': 'subtract_bias',
                    'constant': float(bias)
                }
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
