import torch
from magic import get_splits
from magic_fit import LogisticMLP

def estimate_accuracy(logits: torch.Tensor, label: torch.Tensor)->torch.Tensor:
    logits = logits.round().view(-1)
    acc = torch.eq(logits, label.to(torch.float32)).sum()
    acc = acc /len(label)
    return acc * 100

if __name__ == "__main__":
    x, y = get_splits("train")
    model = LogisticMLP(10, 128)
    logits = model(x)
    acc = estimate_accuracy(logits, y)
    print(f">>>> Accuracy is: {acc:.2f} %")