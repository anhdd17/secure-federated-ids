import torch
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, n_classes: int) -> dict:
    """Accuracy, macro-F1 (each class counts equally: the metric that matters for rare attacks), loss."""
    model.eval()
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, y).item()
    pred = logits.argmax(1)
    cm = torch.bincount(y * n_classes + pred, minlength=n_classes**2).reshape(n_classes, n_classes)
    tp = cm.diag().double()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "acc": (tp.sum() / len(y)).item(),
        "macro_f1": f1.mean().item(),
        "loss": loss,
        "f1_per_class": [round(v, 4) for v in f1.tolist()],
    }
