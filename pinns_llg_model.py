import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class simple_NN(nn.Module): #define neural network
    def __init__(self):
        super(simple_NN, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3),nn.Tanh()
        )

    def forward(self, x):
        out = self.linear_tanh_stack(x)
        return out

#define LSTM model

class LSTMModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim * 3, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, dim * 3)
        self.hidden=self.init_hidden(dim * 3 * 3)
    def init_hidden(self,size):
        return (torch.autograd.Variable(torch.zeros(1, size, 50)).to(device),
                torch.autograd.Variable(torch.zeros(1, size, 50)).to(device))
    def forward(self, x):
        x, self.hidden = self.lstm(x,self.hidden)
        x = self.linear(x)
        return x
