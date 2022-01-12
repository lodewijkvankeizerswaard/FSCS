def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    loss_module = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    best_accuracy = 0
    for epoch in tqdm(range(epochs)):
        for X_batch, labels in train_loader:
            optimizer.zero_grad()

            prediction = model.forward(X_batch.to(device))
            loss = loss_module(prediction, labels.to(device))
            loss.backward()
            optimizer.step()

        val_accuracy = evaluate_model(model, validation_loader, device)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_name)
        
        scheduler.step()

    # Load best model and return it.
    model.load_state_dict(torch.load(checkpoint_name))
    torch.save(model.state_dict(), "models/finished_" + checkpoint_name)

    return model


def num_correct_predictions(predictions, targets):
    predictions = torch.argmax(predictions, axis=1)
    count = (predictions == targets).sum()
    return count.item()


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """

    num_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            predictions = model.forward(X_batch.to(device))
            num_correct += num_correct_predictions(predictions, y_batch.to(device))
            total_samples += len(X_batch)

    avg_accuracy = num_correct / total_samples

    return avg_accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """

    set_seed(seed)
    test_results = defaultdict(list)

    testset = get_test_set(data_dir)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    result = evaluate_model(model, test_loader, device)
    test_results["base"].append(result)

    corruptions = [gaussian_noise_transform,
                   gaussian_blur_transform, contrast_transform, jpeg_transform]

    for func in corruptions:
        for severity in [1, 2, 3, 4, 5]:
            # Load the test set
            testset = get_test_set(data_dir, augmentation=func(severity))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
            result = evaluate_model(model, test_loader, device)
            test_results[func.__name__].append(result)

    return test_results
