from collections import defaultdict

import matplotlib.pyplot as plt

MODEL_METRICS = "/shares/volk.cl.uzh/amoryo/signwriting-animation/models/unconstrained/model/metrics"

if __name__ == "__main__":
    metrics = defaultdict(list)

    with open(MODEL_METRICS, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            for metric in line.strip().split("\t")[1:]:
                name, value = metric.split("=")
                try:
                    metrics[name].append(float(value))
                except ValueError:
                    pass

    for metric in ['perplexity-train']:
        plt.figure(figsize=(10, 5))

        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.plot(metrics[metric])
        # log scale
        plt.yscale('log')
        plt.savefig(f"{metric}.png")
        plt.close()
