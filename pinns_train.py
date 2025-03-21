import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import os
from torch import nn
from pinns_llg_model import simple_NN, LSTMModel
from utils import model_capacity
from pinns_llg_loss import compute_MLFFNN_loss, LSTM_loss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_MLFFNN(training_data, p, domain, num_iter:int=2000, num_coll:int=200):
    train_m , train_t = training_data
    t=torch.sort(domain[-1]*torch.rand(num_coll,dtype=torch.float32))[0].reshape(-1, 1).to(device)
    t.requires_grad=True

    #initializing the model
    model=simple_NN()
    model=model.to(device)
    #printing parameters
    model_capacity(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_list=[]
    # Train
    for ep in range(num_iter):
        model.train()
        optimizer.zero_grad()
        loss = compute_MLFFNN_loss(p, model, t, train_t, train_m,domain)
        loss.backward()
        optimizer.step()
        if loss<0.00001:
            break
        model.eval()
        with torch.no_grad():
            loss_list.append(loss.cpu().detach().numpy())
        if ep % 500 == 0:
            print(f"epoch: {ep}, loss: {loss.item():>7f}")
    return (model, np.array(loss_list))

def train_LSTM(test_train_data, p, domain, n_epochs, dim=3, num_points=1000):
    #get Data
    (_, X_train, X_test, y_train, y_test) = test_train_data

    #initialize model
    model = LSTMModel(dim=dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=3, drop_last=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    loss_list=[]
    dt = torch.tensor((domain[1] - domain[0]) / num_points).to(device)
    for epoch in range(n_epochs):
        model.train()
        
        list=[]
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            model.hidden = model.init_hidden(dim)
            
            y_pred= model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            list.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            model.hidden = model.init_hidden(dim)
            m0 = model(X_train[-3:])
            m0 = m0.detach()
        model.train()
        list1 = []
        for i in range(len(X_test)):
            model.hidden=model.init_hidden(dim)
            # list.append(m0[-1].cpu().detach().numpy())
            optimizer.zero_grad()
            m0 = m0.to(device)
            
            m0 = torch.cat((m0, model(m0)[-1].reshape([1, 3, 3 * dim])), 0)
            loss = loss_fn(LSTM_loss(m0[-1, -2],m0[-1, -1], p, dt, dim), torch.tensor(0.).to(device))
            list1.append(loss.cpu().detach().numpy())
            m0 = m0[1:]
            m0 = m0.detach()
            loss.backward()
            optimizer.step()
        # Validation
        loss_list.append(np.sum(np.array(list))+np.sum(np.array(list1))/(len(list1)+len(list)))
        if epoch % 50 != 0:
            continue
        model.eval()
        with torch.no_grad():
            model.hidden = model.init_hidden(len(X_train))
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train).cpu().detach())
            model.hidden = model.init_hidden(len(X_test))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test).cpu().detach())
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f,loss %.4f" % (epoch, train_rmse, test_rmse,loss_list[-1]))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f,loss %.4f" % (epoch, train_rmse, test_rmse,loss_list[-1]))
    return model, loss_list