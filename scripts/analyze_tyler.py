import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def analyze_experiment(exp_name: str):
    results_dir = Path("results") / exp_name

    if not results_dir.exists():
        print(f"Experiment {exp_name} not found")
        return

    # Collect scores by scale and model
    data = defaultdict(lambda: defaultdict(list))

    for result_file in results_dir.glob("*.json"):
        with open(result_file) as f:
            result = json.load(f)

        # Get scale and model from the JSON data
        scale = result["prompt"].split("/")[-1].replace(".txt", "")
        if result['provider'] == 'anthropic':
            model = '-'.join(result['model'].split('-')[:-1])
        elif result['provider'] == 'openai':
            model = '-'.join(result['model'].split('-')[:2])
        else:
            model = result['model']

        # Get all question scores and average them
        parsed = result["parsed"]
        scores = [v for k, v in parsed.items() if k.isdigit()]
        avg_score = sum(scores) / len(scores) if scores else 0

        data[scale][model].append(avg_score)

    # Get all unique models
    models = set()
    for scale_data in data.values():
        models.update(scale_data.keys())
    models = sorted(models)

    # Print header
    print(f"\n{exp_name}")
    print("=" * 110)
    header = f"{'Scale':<20}" + "".join(f"{m:<20}" for m in models)
    print(header)
    print("-" * 110)

    # Print rows
    for scale in sorted(data.keys()):
        row = f"{scale:<20}"
        for model in models:
            avg = sum(data[scale][model]) / len(data[scale][model]) if data[scale][model] else 0
            row += f"{avg:<20.2f}"
        print(row)

    # Create DataFrame and save
    df_data = {}
    for scale in sorted(data.keys()):
        df_data[scale] = {}
        for model in models:
            avg = sum(data[scale][model]) / len(data[scale][model]) if data[scale][model] else 0
            df_data[scale][model] = avg

    df = pd.DataFrame(df_data).T
    df.index.name = "scale"

    # Save to CSV
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{exp_name}.csv"
    df.to_csv(output_file)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Tyler experiment results")
    parser.add_argument("experiment_name", help="Name of the experiment (e.g., exp0.5_tyler)")
    args = parser.parse_args()

    analyze_experiment(args.experiment_name)
