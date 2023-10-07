import torch
from magic import get_splits, max_eval_iters
from magic_fit import LogisticMLP

x, _ ,_,_= get_splits("train")
in_dim, hidden_dim = x.shape[1], 100
model = LogisticMLP(in_dim = in_dim, hidden_dim = hidden_dim)
model = model

@torch.no_grad()
def _smoothie():

    cache = {}
    losses = torch.zeros(max_eval_iters)
    
    for split in ["train", "valid", "test"]:
        for k in range(max_eval_iters):
            x, y,_,_ = get_splits(split)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy(logits.view(-1), y.to(torch.float32))
            losses[k] = loss.item()
        cache[split] = losses.mean()
    return cache

if __name__ == "__main__":
    cache = _smoothie()
    print(f">>>> train loss: {cache['train']:.4f}, validation loss: {cache['valid']:.4f}, testing loss: {cache['test']:.4f}")




