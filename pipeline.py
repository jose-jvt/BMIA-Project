import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import test_model, train_model

if __name__ == "__main__":
    n_epochs = 100
    batch_size = 32
    resume = True

    model_name = ...
    model = ...

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    criterion = ...  # Replace with your loss function
    metric_fn = ...

    train_dataset = ...
    val_dataset = ...
    test_dataset = ...

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metric_fn=metric_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=n_epochs,
        scheduler=scheduler,
        checkpoint_path="checkpoint.pth",
        resume=resume,
        model_name=model_name
    )

    test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        metric_fn=metric_fn,
        device=device,
    )
