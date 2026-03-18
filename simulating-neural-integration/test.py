import os
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from huggingface_hub import login

def get_final_prompt_activation(model, prompt):
    tokens = model.to_tokens(prompt)
    length = tokens.shape[1]

    num_layers = 26
    with torch.no_grad():
        # Get activations from original model
        _, cache = model.run_with_cache(tokens)

        activations = []
        for layer_idx in range(num_layers):
            # get activation of predicted token
            activations.append(cache[f"blocks.{layer_idx}.hook_resid_post"][:, -1, :])
        # concatenate across layers
        activation = torch.cat(activations, dim=0)

    return activation, length

def activation_steer(model, system_prompt, prompt, layer_idx, max_length, persona_vector, coefficient):
    messages = [
        {"role": "system", "content": f"You are an AI assistant. {system_prompt}"},
        {"role": "user", "content": f"Answer the following question with a few sentences. {prompt}"}
    ]
    full_prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokens = model.to_tokens(full_prompt)
    instruction_length = tokens.shape[1]

    # Generate response token by token
    for _ in range(max_length):
        with torch.no_grad():
            # Define hook to add steering vector to residual stream
            def steering_hook(activations, hook):
                # Add persona vector to the last token position
                activations[:, -1, :] += coefficient * persona_vector[layer_idx]
                return activations

            # Apply hook to residual stream after the specified layer
            hook_name = f"blocks.{layer_idx}.hook_resid_post"

            with model.hooks([(hook_name, steering_hook)]):
                logits = model(tokens)[0, -1, :]

            # Use greedy decoding instead of sampling for more consistent results
            next_token_id = torch.argmax(logits)
            next_token = model.to_string(next_token_id)

            # end generation on creation of new line
            if next_token == "\n" or next_token == "\n\n" or next_token == "\n\n\n":
                break

            tokens = torch.cat([tokens, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

    return model.to_string(tokens[0, instruction_length:])


def main():
    # Login to HuggingFace
    login(token=os.environ.get("HF_TOKEN"))
    torch.manual_seed(42)

    print("Loading model...")
    # Force GPU usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device=device)
    print("Model loaded!\n")

    # Test parameters
    system_prompt = ""
    prompt = "Tell me a short story about a cat."  # Changed to a more creative prompt
    max_length = 100

    # Test with different persona vectors
    persona_vectors_path = Path("persona-vectors/evaluation/stored_persona_vectors")

    # Test only funniness
    test_traits = ["funniness"]
    # Try a range of layers (middle layers often work best)
    test_layers = [15, 20, 25]
    coefficients = [-5.0, 0.0, 5.0]  # Test negative, neutral, and positive steering with even stronger coefficients

    for trait in test_traits:
        vector_path = persona_vectors_path / f"{trait}.pt"
        if not vector_path.exists():
            print(f"Vector not found: {vector_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Testing trait: {trait.upper()}")
        print(f"{'='*80}")

        # Load persona vector directly to GPU
        persona_vector = torch.load(vector_path, weights_only=False, map_location=device)

        print(f"Persona vector shape: {persona_vector.shape}")
        print(f"Persona vector is list of {len(persona_vector)} layer vectors" if isinstance(persona_vector, list) else "")

        for layer_idx in test_layers:
            print(f"\n{'='*80}")
            print(f"Testing Layer: {layer_idx}")
            print(f"{'='*80}")

            for coef in coefficients:
                print(f"\nCoefficient: {coef:+.1f}")
                print("-" * 80)

                response = activation_steer(
                    model=model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    layer_idx=layer_idx,
                    max_length=max_length,
                    persona_vector=persona_vector,
                    coefficient=coef
                )

                print(f"Response: {response}")

    print("\n" + "="*80)
    print("Testing complete!")

if __name__ == "__main__":
    main()
