import torch, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings("ignore")
EPOCHS = 50000
max_eval_iters = 500
eval_iters = 200
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

names = ["fLength", "fWidth", "fSize", "fConc", "fConc1","fAsym","M3Long", "M3Tras","fAlfa", "fDist", "class"]
def read_csv():
    df = pd.read_csv("magic04.data", names = names)
    return df
magic = read_csv()
#print(magic["class"].unique())
magic["class"] = (magic["class"] == "g").astype(int)
#print(magic.head(10))

def explore_covariates():
    for name in names:
        plt.hist(magic[magic["class"] ==1][name], color = "red", density = True, alpha = 0.7, label = "Gamma")
        plt.hist(magic[magic["class"]==0][name], color = "blue", density = True, alpha = 0.7, label = "Hydron")
        plt.legend()
        plt.title(f"Distribution of {name} based on radiation type")
        plt.xlabel(name)
        plt.ylabel("probability")
        plt.show()

def preProcess(data:pd.DataFrame = magic, ros: bool = False):
    X, y = data[data.columns[:-1]].values, data[data.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X) # scaling the covariates
    # Ovarsample for the imbalanced class
    if ros:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    df = np.hstack((X, np.reshape(y, (-1, 1))))
    return X, y,df

BATCH_SIZE = 32

def get_splits(split):

    train, valid, test = np.split(magic.sample(frac = 0.1), [int(0.6 * len(magic)), int(0.8 * len(magic))])
    _,_, train = preProcess(train, True)
    _, _, valid = preProcess(magic, False)
    _, _, test = preProcess(magic, False)

    if split == "train":
        data = train
    elif split == "valid":
        data = valid
    else:
        data = test
    X, y = torch.from_numpy(data[:, :-1]).to(torch.float32), torch.from_numpy(data[:, -1])
    IX = torch.randint(low = 0, high = len(data), size = (BATCH_SIZE,))
    xbatch, ybatch = X[IX], y[IX]

    return xbatch, ybatch, X, y

if __name__ == "__main__":
    xbatch, ybatch,_,_ = get_splits("train")
    print(xbatch, ybatch)
        