import torch
import numpy as np

from pinns_llg_model import simple_NN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#define MLFNN loss

#define dm/dt via torch autograd
def df(f: simple_NN, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = torch.tensor([]).to(device)
    df_value.requires_grad = True
    for i in range(3):
        df_value=torch.cat( (df_value,torch.autograd.grad(
            f(x)[:, i].reshape(len(x), 1),
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]), 1)

    return df_value

#define loss function for PINN
def compute_MLFFNN_loss(p,nn: simple_NN, 
                 t: torch.Tensor = None, 
                 t_train: torch.Tensor = None,
                 m_train: torch.Tensor = None, domain=[0.,30.]
                 ):
    """Compute the full loss function as pde loss + boundary loss + norm_loss +mse_loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    (h_,gamma,alpha)=p
    h = torch.cat((torch.ones(len(t), 1) * h_[0], torch.ones(len(t), 1) * h_[1],torch.ones(len(t),1)*h_[2]), 1).to(device)
    m = nn(t)
    pde_loss = torch.nn.MSELoss()(df(nn, t) + gamma * (torch.linalg.cross(m, h) - alpha * torch.linalg.cross(m, -gamma * torch.linalg.cross(m, h))),
                                  torch.zeros(len(t), 3).to(device))
    norm_loss = torch.nn.MSELoss()((torch.linalg.norm(m, axis=1)).reshape(-1, 1),torch.ones_like(t))
    
    boundary1 = torch.Tensor([[domain[0]]]).to(device)
    boundary2=torch.Tensor([[domain[-1]]]).to(device)

    m_boundary1=torch.tensor([0.0, np.sin((1/180) * np.pi), np.cos((1/180) * np.pi)]).to(device)
    m_boundary2=torch.tensor([0.0,0.,-1.]).to(device)
    
    bc_loss = ((nn(boundary1)[0] - m_boundary1).pow(2).mean()+((nn(boundary2)[0] - m_boundary2).pow(2).mean())).to(device)
    
    mse_loss = torch.nn.MSELoss()(nn(t_train), m_train)

    tot_loss = pde_loss + mse_loss+ norm_loss +bc_loss
    
    return tot_loss

#define LSTM loss

#LSTM loss(pde +norm)
def LSTM_loss(x, m, p, dt, dim):
    (h_, gamma, alpha, J, mat) = p
    pde_loss = torch.zeros(dim, 3)
    norm_loss = torch.zeros(dim)
    m = m.reshape(dim, 3).to(device)
    x = x.reshape(dim, 3).to(device)
    gamma = torch.tensor(gamma,dtype=torch.float).to(device)
    alpha =  torch.tensor(alpha,dtype=torch.float).to(device)
    J = torch.tensor(J,dtype=torch.float).to(device)
    mat = torch.tensor(mat,dtype=torch.float).to(device)
    h = torch.tensor([h_], dtype=torch.float).reshape(3).to(device)

    
    for i in range(dim):
        pde_loss[i] = (m[i] - x[i]) / dt + gamma * \
        (torch.linalg.cross(m[i], h + J * torch.matmul(mat[i], m)) - \
          alpha * torch.linalg.cross(m[i], -gamma * \
                                     torch.linalg.cross(m[i], h + J * torch.matmul(mat[i], m))))
        norm_loss[i] = (torch.tensor(1.) - torch.linalg.norm(m[i])) ** 2

    pde_loss=torch.linalg.norm(pde_loss)
    return torch.sum(pde_loss + norm_loss).to(device)
