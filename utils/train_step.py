import torch


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Perform a single training step on the provided model using the given data.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the training data.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        device (torch.device): The device (CPU or GPU) on which to perform the training.

    Returns:
        tuple: average training loss and accuracy for the current step.
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        target_label = torch.argmax(y, dim=1)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == target_label).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
