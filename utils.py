import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pinns_llg_model import simple_NN
from pinns_llg_model import LSTMModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#prints number of layers and trainable parameters in the MLFFNN model
def model_capacity(model:simple_NN):
    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

#define error of model ouptut wrt numerical solution (MLFFNN)
def error(model:simple_NN,t_,result_odeint):
    model.eval()
    with torch.no_grad():
        t_=torch.from_numpy(np.float32(t_)).reshape(-1, 1).to(device)
        f_eval = (model(t_)).to('cpu')
        f_eval=f_eval.detach().numpy()
        err=(np.mean(np.linalg.norm(result_odeint-f_eval,axis=1)**2))**(0.5)
    return err

#plotting of model output and training data(MLFFNN)
def plot_MLFFNN(model:simple_NN, train_t, train_m, domain):
    model.eval()
    with torch.no_grad():
        t_eval = torch.linspace(domain[0], domain[1], steps=1000).reshape(-1, 1)
        f_eval = (model(t_eval.to(device))).to('cpu')
        # plotting
        train_t1=train_t.cpu().detach().numpy()
        train_m1=train_m.cpu().detach().numpy()
        
        #2D plot of mx,my,mz vs time
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        ax.scatter(train_t1, train_m1[:,0], label="Training data", color="#1f77b4")
        ax.plot(t_eval.detach().numpy(), f_eval.detach()[:,0].numpy(), label="NN approximation", color="#1f77b4")
        
        ax.scatter(train_t1, train_m1[:,1], label="Training data", color="#ff7f0e")
        ax.plot(t_eval.detach().numpy(), f_eval.detach()[:,1].numpy(), label="NN approximation", color="#ff7f0e")
        
        ax.scatter(train_t1, train_m1[:,2], label="Training data", color="#2ca02c")
        ax.plot(t_eval.detach().numpy(), f_eval.detach()[:,2].numpy(), label="NN approximation", color="#2ca02c")
        
        ax.set(title="Neural Network Regression", xlabel="x", ylabel="y")
        ax.legend()
        plt.show()
        
        #3D plot of (mx,my,mz)
        figure = plt.figure(figsize=(8, 8))
        bx = figure.add_subplot(2, 1, 2, projection='3d')
        bx.scatter(train_m1[:, 0],
                train_m1[:, 1],
                train_m1[:, 2], color="#1f77b4")
        bx.plot(f_eval.detach().numpy()[:,0],
                   f_eval.detach().numpy()[:,1],
                   f_eval.detach().numpy()[:,2], color="#1f77b4")
        bx.set_title("Magnetisation for 0D single spin")
        plt.show()

#define plotting for LSTM
xyz=("x",'y','z')
def plot_LSTM(model:LSTMModel, timeseries, X_train, X_test, lookback, dim=3):
    train_size = len(X_train) + lookback
    model.to(device)
    model.eval()
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        model.hidden = model.init_hidden(len(X_train))
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        model.hidden = model.init_hidden(len(X_train))
        train_plot[lookback:train_size] = model(X_train)[:,  -1, :].cpu().detach()
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        model.hidden = model.init_hidden(len(X_test))
        test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :].cpu().detach()
        plt.figure(figsize=(18, 12))
        for j in range(dim):
            for i in range(3):
                plt.subplot(dim, 3, 3 * j + i + 1)
            
                plt.plot(timeseries[:, i], label='true')
                plt.plot(train_plot[:, i], c ='r', label='training')
                plt.plot(test_plot[:, i], c ='g', label='prediction')
                plt.legend()
                plt.title("Spin "+str(j + 1) + ", " + str(xyz[i]))
        plt.show()
        figure = plt.figure(figsize=(18, 12))

        for j in range(dim):
            bx = figure.add_subplot(dim, 3, j+1, projection='3d')
            bx.plot(timeseries[:, 3 * j],timeseries[:,3 * j + 1],timeseries[:, 3 * j + 2], label='true')
            bx.plot(train_plot[:, 3 * j],train_plot[:,3 * j + 1],train_plot[:, 3 * j + 2], c='r',label='training')
            bx.plot(test_plot[:, 3 * j],test_plot[:, 3 * j + 1],test_plot[:,3 * j + 2], c='g',label='prediction')
            bx.legend()
            plt.title("Spin "+str(j+1))
        plt.show()

#defines the interactions between spins
def linear_interaction_matrix(dim):
    mat = [[0]*dim for _ in range(dim)] 
    for i in range(dim):
        if(i - 1 >= 0):
            mat[i][i-1] = 1
        if(i + 1 < dim):
            mat[i][i+1] = 1
    return mat