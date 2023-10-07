import torch
from magic import get_splits, BATCH_SIZE

class LogisticMLP:

    def __init__(self, in_dim, hidden_dim)->None:
        self.weights = torch.randn(size = (in_dim, hidden_dim)) * (2 / in_dim)**-0.5
        self.bias = torch.zeros(hidden_dim)
        self.logits = torch.randn(size = (hidden_dim, 1))
        self.b_out = torch.randn(1)
    
    def __call__(self, x: torch.Tensor)->torch.Tensor:
        out = x @ self.weights + self.bias
        out = torch.nn.functional.relu(out)
        out = out @ self.logits + self.b_out
        out = torch.nn.functional.sigmoid(out)
        return out
    
    def parameters(self):
        return [self.weights, self.bias, self.logits,self.b_out]
    
    def gradients(self):
        return [g.retain_grad() for g in [self.weights, self.bias, self.logits, self.b_out]]

if __name__ == "__main__":
    x, y = get_splits("train")
    in_dim, hidden_dim = x.shape[1], 100
    model = LogisticMLP(in_dim, hidden_dim)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 1)
