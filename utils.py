import torch
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt
import os
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

def save_checkpoint(path, model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: Callable,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    checkpoint_path: str = "checkpoint.pth",
    resume: bool = False,
    model_name: str = "model"
):
    model.to(device)

    timestamp = get_timestamp()
    best_model_path = f"best_model_{model_name}_{timestamp}.pth"
    best_model_full_path = f"best_model_{model_name}_{timestamp}_full.pth"

    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resumed from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            score = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                score += metric_fn(outputs, labels)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_score = score / total
            train_losses.append(epoch_loss)

            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Score: {epoch_score:.4f}')

            if val_loader:
                model.eval()
                val_running_loss = 0.0
                val_score = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        val_score += metric_fn(outputs, labels)
                        val_total += labels.size(0)

                avg_val_loss = val_running_loss / val_total
                avg_val_score = val_score / val_total
                val_losses.append(avg_val_loss)

                print(f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_score:.4f}')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_path)
                    torch.save(model, best_model_full_path)
                    print(f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
                    print(f"Saved: {best_model_path} and {best_model_full_path}")

            if scheduler:
                scheduler.step()

            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss)
        print(f"Checkpoint saved to {checkpoint_path}. Exiting training.")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    if val_loader and val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: Callable,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates a trained PyTorch model on a test dataset using a custom metric.

    Parameters:
    - model: the trained PyTorch model
    - test_loader: DataLoader for the test data
    - criterion: loss function (e.g., nn.CrossEntropyLoss)
    - metric_fn: a callable that takes (predictions, labels) and returns a metric value
    - device: torch.device (e.g., torch.device('cuda') or torch.device('cpu'))

    Returns:
    - test_loss: average test loss
    - test_metric: value of the provided metric function
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_loss = total_loss / len(test_loader.dataset)
    test_metric = metric_fn(all_preds, all_labels)

    print(f"Test Loss: {test_loss:.4f}, Test Metric: {test_metric:.4f}")
    return test_loss, test_metric
