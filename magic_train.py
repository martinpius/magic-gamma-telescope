from magic_fit import LogisticMLP
from magic_smooth import _smoothie
from magic import get_splits, eval_iters, EPOCHS
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
from accuracy import estimate_accuracy
from report import get_report

model = LogisticMLP(in_dim = 10, hidden_dim = 128)
for p in model.parameters(): p.requires_grad = True
optimizer = torch.optim.AdamW(params = model.parameters(), lr = 0.01)
print(f">>>> {model.__class__.__name__} has {sum([p.numel() for p in model.parameters()]):,} trainable parameters\n")
for epoch in tqdm(range(EPOCHS)):
    if epoch % eval_iters == 0:
        cache = _smoothie()
        # print(f">>>> Epoch: {epoch + 1 if epoch == 0 else epoch},\
        #       Acc on train: {cache['train']:.4f},\
        #         Eval on the validation set: {cache['valid']:.4f},\
        #             Eval on the test set: {cache['test']:.4f}")
    
    xb, yb, X, y = get_splits("train")
    logits = model(xb)
    loss = F.binary_cross_entropy(logits.view(-1), yb.to(torch.float32))
    acc = estimate_accuracy(logits, yb)
    for p in model.parameters(): p.grad = None
    #params = model.parameters()
    #grads = [p.retain_grad() for p in params]
    #optimizer.zero_grad()
    loss.backward()
    LR = 0.01 if epoch % 35000 == 0 else 0.001
    for p in model.parameters():
        p.data -= LR * p.grad
    #optimizer.step()
    if epoch % 1000 == 0:
        print(f">>>> Epoch: {epoch + 1 if epoch == 0 else epoch}, Train Loss: {loss.item():.4f}, Train accuracy: {acc.item():.2f} %")

r1, r2 = get_report(model)
print(f">>>> Classification report on validation set:\n {r1}\n\n>>>> Classification report on test set: {r2}")
    
