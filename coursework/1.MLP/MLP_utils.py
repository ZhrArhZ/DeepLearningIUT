import torch
import matplotlib.pyplot as plt


def eval_model(model, criterion, val_loader, device):
    """Evaluate model on val_loader.
    Returns (avg_loss, accuracy) for validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            x = x.reshape(x.shape[0], -1)

            logits = model(x)

            loss = criterion(logits, y)
            running_loss += loss.item() * y.shape[0]

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.shape[0]

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def test_model(model, test_loader, device):
    """Evaluate model on test_loader. Returns (y_pred, y_true) for test set"""
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            x = x.reshape(x.shape[0], -1)

            logits = model(x)
            preds = logits.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    return y_pred, y_true


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    scheduler=None,
    scheduler_step_per_epoch=True,
    print_every=1,
    pr=False
):
    """
    Train model for num_epochs, evaluating on val_loader at end of each
    epoch.

    Returns dict with loss/accuracy history.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    model.to(device)

    val_loss_best = float('inf')
    patience_threshold = 3
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device=device, non_blocking=True)
            y_batch = y_batch.to(device=device, non_blocking=True)
            x_batch = x_batch.reshape(x_batch.shape[0], -1)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_batch.shape[0]
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.shape[0]

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = eval_model(model, criterion, val_loader, device)

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= patience_threshold:
                break

        if scheduler is not None:
            if scheduler_step_per_epoch:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % print_every == 0 and pr:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return history


def plot_metric(train_metric, val_metric, label):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(train_metric, label=f"Train {label}")
    ax.plot(val_metric, label=f"Validation {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(f"Training and Validation {label}")
    ax.legend()
    plt.tight_layout()
    plt.show()
