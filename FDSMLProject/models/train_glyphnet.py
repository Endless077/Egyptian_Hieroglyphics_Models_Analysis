import utils

@hydra.main(config_path="../configs", config_name="config_glyphnet")
def main_Glyphnet(cfg):
    """
    Funzione principale per l'addestramento e la valutazione del modello Glyphnet.

    Args:
        cfg (DictConfig): Configurazione di Hydra contenente i percorsi dei dati, parametri di addestramento, ecc.

    Returns:
        Dict[str, List[float]]: Metriche raccolte durante l'addestramento e la validazione.
    """
    # Avvio del tracker di CodeCarbon per monitorare le emissioni
    tracker = EmissionsTracker()  # Puoi specificare una directory di output se necessario
    tracker.start()

    train_labels, test_path, train_path = load_and_split_data(cfg)

    train_set = GlyphData(root=train_path, class_to_idx=train_labels, transform=train_transform)

    logging.info("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)),
        train_set.targets,
        test_size=cfg.data.val_fraction,
        shuffle=True,
        random_state=cfg.model.seed
    )

    logging.info(f"CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Passare la classe del modello (Glyphnet) come argomento
    best_params, grid_search_results, best_optimizer_state, metrics = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, Glyphnet)

    # Verifica se metrics è None
    if metrics is None:
        logging.error("Metrics were not returned from perform_grid_search, cannot proceed.")
        return

    # Dopo la Grid Search: carica il miglior modello e valuta
    test_labels_set = {l for l in [p.strip("/") for p in listdir(test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(root=test_path, class_to_idx=test_labels,
                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                       transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    # Carica il miglior modello trovato durante la Grid Search
    model = Glyphnet(num_classes=len(train_labels))
    model.load_state_dict(torch.load("Glyphnet_best_model_weights.pth"))
    model.to(device)

    logging.info("Checking quality on test set:")
    evaluate_model(model, test_loader, device, num_classes=len(train_labels))

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), "checkpoint.bin")

    logging.info("Saving the optimizer state.")
    torch.save(best_optimizer_state, "optimizer_state.pth")

    plot_grid_search_results(grid_search_results, param_grid)
    tracker.stop()

    plot_learning_curves(metrics)

    return metrics
    
    if __name__ == "__main__":
    metrics = main_Glyphnet()