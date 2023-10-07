from sklearn.metrics import classification_report
from magic import get_splits

def get_report(model):
     _,_, x_val, y_val = get_splits("valid")
     _,_,x_ts, y_ts = get_splits("test")
     logits_val = model(x_val)
     logits_ts = model(x_ts)
     rep1 = classification_report(y_val.detach().numpy(), logits_val.view(-1).round().detach().numpy())
     rep2 = classification_report(y_ts.detach().numpy(), logits_ts.view(-1).round().detach().numpy())
     return rep1, rep2

