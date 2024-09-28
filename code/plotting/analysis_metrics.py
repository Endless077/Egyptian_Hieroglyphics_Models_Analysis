# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Custom colors for consistency
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Model Data
models = ['TResNet', 'ATCNet', 'Glyphnet', 'CombinedTResNet']

# Performance Metrics
accuracy = [79.54, 0.50, 4.26, 80.5]
f1_score = [78.69, 0.05, 0.19, 79.53]
precision = [79.61, 0.03, 0.20, 81.25]
recall = [79.54, 0.16, 1.00, 80.50]

# Energy Metrics
gpu_energy = [0.8460, 0.2388, 0.2366, 0.7016]
cpu_energy = [0.3128, 0.1135, 0.1114, 0.2552]
ram_energy = [0.0859, 0.0312, 0.0306, 0.0701]
total_energy = [1.2448, 0.3834, 0.3786, 1.0270]

# Training Time (in seconds)
training_time = [26497.82, 9610.30, 9438.43, 21621.20]


def plot_performance_metrics(models: list[str], 
                             accuracy: list[float], 
                             f1_score: list[float], 
                             precision: list[float], 
                             recall: list[float]) -> None:
    """
    Creates a bar chart to compare performance metrics (Accuracy, F1-Score, Precision, Recall)
    across different models.

    Args:
        models (list): A list of model names.
        accuracy (list): A list of accuracy values for each model.
        f1_score (list): A list of F1-Score values for each model.
        precision (list): A list of precision values for each model.
        recall (list): A list of recall values for each model.

    Returns:
        None
    """
    x = np.arange(len(models))
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars for each metric
    rects1 = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy', color=colors[0])
    rects2 = ax.bar(x - width / 2, f1_score, width, label='F1-Score', color=colors[1])
    rects3 = ax.bar(x + width / 2, precision, width, label='Precision', color=colors[2])
    rects4 = ax.bar(x + width * 1.5, recall, width, label='Recall', color=colors[3])

    # Add labels above the bars
    for rects in [rects1, rects2, rects3, rects4]:
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    ax.set_yscale('log')
    ax.set_ylim(0.01, 160)

    ax.set_xlabel('Models')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Comparison of Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_energy_consumption(models: list[str], 
                            gpu_energy: list[float], 
                            cpu_energy: list[float], 
                            ram_energy: list[float]) -> None:
    """
    Creates a bar chart to compare energy consumption (GPU, CPU, RAM) across different models.

    Args:
        models (list): A list of model names.
        gpu_energy (list): A list of GPU energy consumption values for each model.
        cpu_energy (list): A list of CPU energy consumption values for each model.
        ram_energy (list): A list of RAM energy consumption values for each model.

    Returns:
        None
    """
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, gpu_energy, width, label='GPU Energy (kWh)', color=colors[0])
    rects2 = ax.bar(x, cpu_energy, width, label='CPU Energy (kWh)', color=colors[1])
    rects3 = ax.bar(x + width, ram_energy, width, label='RAM Energy (kWh)', color=colors[2])

    ax.set_xlabel('Models')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('Energy Consumption Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_training_time(models: list[str], training_time: list[float]) -> None:
    """
    Creates a bar chart to compare training time across different models.

    Args:
        models (list): A list of model names.
        training_time (list): A list of training time values (in seconds) for each model.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(models, training_time, color=colors[0])

    ax.set_xlabel('Models')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison')

    fig.tight_layout()
    plt.show()


def plot_grid_search_results(results: dict[float, dict[int, float]]) -> None:
    """
    Creates a line chart to visualize grid search results in terms of validation accuracy
    for different batch sizes and learning rates.

    Args:
        results (dict): A dictionary containing grid search results, with learning rates as keys
                        and another dictionary of batch sizes and accuracies as values.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    for lr, batches in results.items():
        batch_sizes = list(batches.keys())
        accuracies = list(batches.values())
        plt.plot(batch_sizes, accuracies, marker='o', label=f"Learning Rate: {lr}")

    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Grid Search Results: Validation Accuracy by Batch Size and Learning Rate')
    plt.legend(title="Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Visualize the plots
plot_performance_metrics(models, accuracy, f1_score, precision, recall)
plot_energy_consumption(models, gpu_energy, cpu_energy, ram_energy)
plot_training_time(models, training_time)

# Data for the grid search plot
results = {
    0.0001: {16: 0.7954, 32: 0.7929, 64: 0.7882},
    0.001: {16: 0.7597, 32: 0.7747, 64: 0.7802}
}

plot_grid_search_results(results)
