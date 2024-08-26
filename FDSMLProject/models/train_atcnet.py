import utils

@hydra.main(config_path="../configs", config_name="config_atcnet")
def main_ATCNet(cfg):
    """
    Funzione principale per l'addestramento e la valutazione del modello ATCNet.

    Args:
        cfg (DictConfig): Configurazione di Hydra contenente i percorsi dei dati, parametri di addestramento, ecc.

    Returns:
        Dict[str, List[float]]: Metriche raccolte durante l'addestramento e la validazione.
    """
    # Avvio del tracker di CodeCarbon per monitorare le emissioni
    tracker = EmissionsTracker()  # Puoi specificare una directory di output se necessario
    tracker.start()

    train_labels, test_path, train_path = load_and_split_data(cfg)

    train_set = datasets.ImageFolder(root=train_path, transform=train_transform)

    logging.info("Splitting data...")

    train_indices, val_indices = train_test_split(range(len(train_set)), test_size=cfg.data.val_fraction,
                                                  random_state=cfg.model.seed)
    logging.info(f"CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Passare la classe del modello (ATCNet) come argomento
    best_params, grid_search_results, best_optimizer_state, metrics = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, ATCNet)

    # Verifica se metrics è None
    if metrics is None:
        logging.error("Metrics were not returned from perform_grid_search, cannot proceed.")
        return

    test_set = datasets.ImageFolder(root=test_path, transform=train_transform)
    test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    model = ATCNet(n_classes=len(train_set.classes))

    # Caricamento pesi specifici per ATCNet
    model.load_state_dict(torch.load("ATCNet_best_model_weights.pth"))
    model.to(device)

    logging.info("Checking quality on test set:")
    evaluate_model(model, test_loader, device, num_classes=len(train_set.classes))

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), "ATCNet_checkpoint.bin")

    plot_grid_search_results(grid_search_results, param_grid)

    tracker.stop()

    # Plot delle curve di apprendimento
    plot_learning_curves(metrics)

    return metrics
    
    if __name__ == "__main__":
    metrics = main_ATCNet()