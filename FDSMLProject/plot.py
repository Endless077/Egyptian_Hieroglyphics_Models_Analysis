import matplotlib.pyplot as plt
import numpy as np

# Dati dei modelli
models = ['TResNet', 'ATCNet', 'Glyphnet', 'CombinedTResNet']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colori personalizzati coerenti

# Metriche di Performance
accuracy = [79.54, 0.50, 4.26, 80.5]
f1_score = [78.69, 0.05, 0.19, 79.53]
precision = [79.61, 0.03, 0.20, 81.25]
recall = [79.54, 0.16, 1.00, 80.50]

# Metriche Energetiche
gpu_energy = [0.8460, 0.2388, 0.2366, 0.7016]
cpu_energy = [0.3128, 0.1135, 0.1114, 0.2552]
ram_energy = [0.0859, 0.0312, 0.0306, 0.0701]
total_energy = [1.2448, 0.3834, 0.3786, 1.0270]

# Tempo di Addestramento (in secondi)
training_time = [26497.82, 9610.30, 9438.43, 21621.20]


def plot_performance_metrics(models, accuracy, f1_score, precision, recall):
    """
    Crea un grafico a barre per confrontare le metriche di performance (Accuracy, F1-Score, Precision, Recall)
    tra diversi modelli.

    Args:
        models (list): Una lista di nomi dei modelli.
        accuracy (list): Una lista di valori di accuracy per ciascun modello.
        f1_score (list): Una lista di valori di F1-Score per ciascun modello.
        precision (list): Una lista di valori di precision per ciascun modello.
        recall (list): Una lista di valori di recall per ciascun modello.

    Returns:
        None
    """
    x = np.arange(len(models))
    width = 0.2  # Larghezza delle barre

    fig, ax = plt.subplots(figsize=(10, 6))

    # Creazione delle barre per ogni metrica
    rects1 = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x - width / 2, f1_score, width, label='F1-Score', color='#ff7f0e')
    rects3 = ax.bar(x + width / 2, precision, width, label='Precision', color='#2ca02c')
    rects4 = ax.bar(x + width * 1.5, recall, width, label='Recall', color='#d62728')

    # Aggiunta delle etichette sopra le barre
    for rects in [rects1, rects2, rects3, rects4]:
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    ax.set_yscale('log')
    ax.set_ylim(0.01, 160)

    ax.set_xlabel('Modelli')
    ax.set_ylabel('Percentuale (%)')
    ax.set_title('Confronto delle Metriche di Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_energy_consumption(models, gpu_energy, cpu_energy, ram_energy):
    """
    Crea un grafico a barre per confrontare il consumo energetico (GPU, CPU, RAM) tra diversi modelli.

    Args:
        models (list): Una lista di nomi dei modelli.
        gpu_energy (list): Una lista di valori di consumo energetico della GPU per ciascun modello.
        cpu_energy (list): Una lista di valori di consumo energetico della CPU per ciascun modello.
        ram_energy (list): Una lista di valori di consumo energetico della RAM per ciascun modello.

    Returns:
        None
    """
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, gpu_energy, width, label='Energia GPU (kWh)', color='#1f77b4')
    rects2 = ax.bar(x, cpu_energy, width, label='Energia CPU (kWh)', color='#ff7f0e')
    rects3 = ax.bar(x + width, ram_energy, width, label='Energia RAM (kWh)', color='#2ca02c')

    ax.set_xlabel('Modelli')
    ax.set_ylabel('Consumo Energetico (kWh)')
    ax.set_title('Confronto del Consumo Energetico')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_training_time(models, training_time):
    """
    Crea un grafico a barre per confrontare il tempo di addestramento tra diversi modelli.

    Args:
        models (list): Una lista di nomi dei modelli.
        training_time (list): Una lista di valori di tempo di addestramento (in secondi) per ciascun modello.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(models, training_time, color='#1f77b4')

    ax.set_xlabel('Modelli')
    ax.set_ylabel('Tempo di Addestramento (s)')
    ax.set_title('Confronto del Tempo di Addestramento')

    fig.tight_layout()
    plt.show()


def plot_grid_search_results(results):
    """
    Crea un grafico a linee per visualizzare i risultati di una grid search in termini di accuratezza di validazione
    per diversi batch size e learning rate.

    Args:
        results (dict): Un dizionario contenente i risultati della grid search, con i learning rate come chiavi
                        e un altro dizionario di batch size e accuratezza come valori.

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


# Visualizza i grafici
plot_performance_metrics(models, accuracy, f1_score, precision, recall)
plot_energy_consumption(models, gpu_energy, cpu_energy, ram_energy)
plot_training_time(models, training_time)

# Dati per il grafico della grid search
results = {
    0.0001: {16: 0.7954, 32: 0.7929, 64: 0.7882},
    0.001: {16: 0.7597, 32: 0.7747, 64: 0.7802}
}

plot_grid_search_results(results)
