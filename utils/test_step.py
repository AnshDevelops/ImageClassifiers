import torch


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """
    Perform a single evaluation step on the provided model using the given data.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        device (torch.device): The device (CPU or GPU) on which to perform the evaluation.

    Returns:
        tuple:average evaluation loss and accuracy for the current step.
    """

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            target_label = torch.argmax(y, dim=1)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == target_label).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
