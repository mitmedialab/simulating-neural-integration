import json
from pathlib import Path
import numpy as np

def load_eval_file(filepath):
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_averages(eval_data):
    """Calculate average scores for first 10 (deceptive) and last 10 (honest) scenarios"""
    # First 10 scenarios are deceptive
    deceptive_avgs = [item['avg'] for item in eval_data[:10] if item['avg'] is not None]

    # Last 10 scenarios are honest
    honest_avgs = [item['avg'] for item in eval_data[10:20] if item['avg'] is not None]

    deceptive_avg = np.mean(deceptive_avgs) if deceptive_avgs else None
    honest_avg = np.mean(honest_avgs) if honest_avgs else None

    return {
        "deceptive": deceptive_avg,
        "honest": honest_avg
    }

def main():
    # Paths to evaluation files
    eval_dir = Path("scripts/eval")

    control_file = eval_dir / "control.json"
    steer_file = eval_dir / "1.json"
    reverse_file = eval_dir / "2.json"

    print("="*80)
    print("SUMMARIZING EVALUATION RESULTS")
    print("="*80)

    summary = {}

    # Process control
    if control_file.exists():
        print("\nProcessing control...")
        control_data = load_eval_file(control_file)
        summary["control"] = calculate_averages(control_data)
        print(f"  Deceptive prompts avg: {summary['control']['deceptive']:.3f}")
        print(f"  Honest prompts avg: {summary['control']['honest']:.3f}")
    else:
        print(f"\nWarning: {control_file} not found")
        summary["control"] = {"deceptive": None, "honest": None}

    # Process steer (experiment 1)
    if steer_file.exists():
        print("\nProcessing steer (same direction)...")
        steer_data = load_eval_file(steer_file)
        summary["steer"] = calculate_averages(steer_data)
        print(f"  Deceptive prompts avg: {summary['steer']['deceptive']:.3f}")
        print(f"  Honest prompts avg: {summary['steer']['honest']:.3f}")
    else:
        print(f"\nWarning: {steer_file} not found")
        summary["steer"] = {"deceptive": None, "honest": None}

    # Process reverse (experiment 2)
    if reverse_file.exists():
        print("\nProcessing reverse (inverse direction)...")
        reverse_data = load_eval_file(reverse_file)
        summary["reverse"] = calculate_averages(reverse_data)
        print(f"  Deceptive prompts avg: {summary['reverse']['deceptive']:.3f}")
        print(f"  Honest prompts avg: {summary['reverse']['honest']:.3f}")
    else:
        print(f"\nWarning: {reverse_file} not found")
        summary["reverse"] = {"deceptive": None, "honest": None}

    # Save summary
    output_file = eval_dir / "sum.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print(f"Summary saved to {output_file}")
    print("="*80)

    # Print formatted summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nControl (no steering):")
    print(f"  Deceptive scenarios: {summary['control']['deceptive']:.3f}")
    print(f"  Honest scenarios: {summary['control']['honest']:.3f}")

    print("\nSteer (same direction):")
    print(f"  Deceptive scenarios: {summary['steer']['deceptive']:.3f}")
    print(f"  Honest scenarios: {summary['steer']['honest']:.3f}")

    print("\nReverse (inverse direction):")
    print(f"  Deceptive scenarios: {summary['reverse']['deceptive']:.3f}")
    print(f"  Honest scenarios: {summary['reverse']['honest']:.3f}")

if __name__ == "__main__":
    main()
