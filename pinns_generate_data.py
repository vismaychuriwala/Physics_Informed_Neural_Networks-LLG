import torch
import numpy as np
from scipy.integrate import odeint

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#generate training data for hyperparameters: (h_, gamma, alpha) = p
def generate_data_MLFFNN(p, domain, m0, num_train:int=20, num_points=1000):

    def func(t, m, h, gamma, alpha):
     dm = -gamma * (np.cross(m, h) - alpha * np.cross(m, -gamma * np.cross(m, h)))
     return dm

    t_= np.linspace(domain[0], domain[1], num=num_points)
    #generating training data for given parameters p 
    result_odeint = odeint(func, m0, t_, p, tfirst=True)
    train_m = []
    train_t = []
    train_list = np.linspace(0, num_points, num=num_train + 2)[1:-1].astype(int)
    for i in train_list:
        train_m.append(result_odeint[i])
        train_t.append([t_[i]])
    
    train_m=torch.tensor(np.array(train_m), dtype=torch.float32, requires_grad=True).to(device)
    train_t=torch.tensor(np.array(train_t), dtype=torch.float32, requires_grad=True).to(device)
    return (train_m, train_t)

#prepare training and test datasets for LSTM

def generate_data_LSTM(p, domain, dim =3, train_test_ratio = 0.5, num_points=1000, lookback=3):

    #defining the interaction matrix between spins
    def func(t, m, h, gamma, alpha, J, mat):
        m = m.reshape(dim, 3)
        dm = np.ones_like(m)
        for i in range(dim):
            dm[i] =- gamma * (np.cross(m[i], h + J * np.matmul(mat[i], m)) - \
                          alpha * np.cross(m[i], -gamma * np.cross(m[i], h + J * np.matmul(mat[i], m))))
        return dm.flatten()

    #define inital magentisation
    m0 = [0, np.sin((1/180)*np.pi), np.cos((1/180)*np.pi)] 
    m1 = [0, -np.sin((1/180)*np.pi), -np.cos((1/180)*np.pi)]
    m = [m0] * dim
    for i in range(dim):
        if (i // 2 != 0):
            m[i] = m1
    m = np.array(m).flatten()
    t_ = np.linspace(domain[0], domain[1], num=num_points)
    result_odeint = odeint(func, m, t_, p, tfirst=True).astype(np.float32)

    timeseries = result_odeint.astype('float32')
    # train-test split for time series
    train_size = int(len(timeseries) * train_test_ratio)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    def create_dataset(dataset, lookback):
        X, y = [], []
        for i in range(len(dataset) - lookback):
            feature = dataset[i:i + lookback]
            target = dataset[i + 1:i + lookback + 1]
            X.append(feature)
            y.append(target)
        return torch.tensor(X), torch.tensor(y)
    
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    X_train=X_train.to(device)
    y_train=y_train.to(device)
    X_test=X_test.to(device)
    y_test=y_test.to(device)
    return (timeseries, X_train, X_test, y_train, y_test)