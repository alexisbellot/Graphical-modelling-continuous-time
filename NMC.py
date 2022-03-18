from source.locally_connected import LocallyConnected
import source.utils as utils
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F 
from torchdiffeq import odeint_adjoint as odeint
    
    
class NNODEF(nn.Module):
    def __init__(self, in_dim, hid_dim, time_invariant=True):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):

        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

class MLPODEF(nn.Module):
    def __init__(self, dims, GL_reg=0.01, bias=True, time_invariant=True): 
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super(MLPODEF, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg # adaptive lasso parameter

        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0]+1, dims[0] * dims[1], bias=bias)
            
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(dims[0], dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
            
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = fc(self.elu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x = x.unsqueeze(dim=1) # [n, 1, d]
        return x

    def l2_reg(self):
        '''L2 regularization on all parameters'''
        reg = 0.
        fc1_weight = self.fc1.weight # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_reg(self):
        '''L1 regularization on input layer parameters'''
        return torch.sum(torch.abs(self.fc1.weight))
    
    def group_weights(self, gamma = .5):
        '''Group lasso weights'''
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight ** 2, dim=1).pow(gamma).data  # [i, j]
        return weights
        
    def causal_graph(self, w_threshold=0.3):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight ** 2, dim=1).pow(0.5)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        W[np.abs(W) < w_threshold] = 0
        return np.round(W,2)
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()
            
            
def train(func, data, n_steps, times = None, plot_freq=10, horizon=5, l1_reg=0, l2_reg=0, plot=True, irregular=False):
    '''Train Neural ODE
    
    func: nn.Module class
    data: tensor of shape [number of trajectories, 1, number of variables]
    n_steps (float): number of training steps
    plot_freq (int): result plotting frequency
    horizon (int): prediction horizon
    l1_reg (float): L1 regularization strength
    l2_reg (float): L1 regularization strength'''
    
    batch_time = horizon
    data_size = data.shape[0]
    device = torch.device('cpu')
    
    if not irregular:
        times = np.linspace(0, data.shape[0], data.shape[0])
        times_np = np.hstack([times[:, None]])
        times = torch.from_numpy(times_np[:, :, None])

    def create_batch(batch_size):
        s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64),  batch_size, 
                                              replace=False))
        batch_y0 = data[s]  # (M, D)
        batch_t = times[:batch_time].squeeze()  # (T)
        if irregular:
            #batch_t = torch.cat([times[time:time+batch_time, :, :] for time in s],1).squeeze()  
            batch_t = times[s:s+batch_time].squeeze() - times[s].squeeze()  # (T)
        batch_y = torch.stack([data[s + i] for i in range(batch_time)], dim=0)
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

    def proximal(w, lam=0.1, eta=0.1):
        '''Proximal step'''
        # w shape [j * m1, i]
        wadj = w.view(func.dims[0], -1, func.dims[0])  # [j, m1, i]
        tmp = torch.sum(wadj ** 2, dim=1).pow(0.5) - lam*eta 
        alpha = torch.clamp(tmp, min=0)
        v = torch.nn.functional.normalize(wadj, dim=1)*alpha[:,None,:]
        w.data = v.view(-1,func.dims[0])
    
    lr = 0.005
    optimizer = torch.optim.Adam(func.parameters(), lr=lr)
    
    for i in range(n_steps):
                
        if irregular:
            obs0_, ts_, obs_ = create_batch(batch_size = 1) 
            z_ = odeint(func, obs0_, ts_)
            loss = F.mse_loss(z_, obs_.detach())
                
            #loss = 0
            #for _ in range(20):
            #    obs0_, ts_, obs_ = create_batch(batch_size = 1) 
            #    z_ = odeint(func, obs0_, ts_)
            #    loss += F.mse_loss(z_, obs_.detach()) / 20
                
        else:    
            obs0_, ts_, obs_ = create_batch(batch_size = 20)
            z_ = odeint(func, obs0_, ts_)
            loss = F.mse_loss(z_, obs_.detach())
            
        if l2_reg != 0:
            loss = loss + l2_reg * func.l2_reg()
        if l1_reg != 0:
            loss = loss + l1_reg * func.fc1_reg()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        proximal(func.fc1.weight, lam=func.GL_reg, eta=0.01)
        
        if i == 2000:
            print('Updating learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr*0.5
                
        
        if plot and i % plot_freq == 0:
            z_p = odeint(func, data[0], times[:100].squeeze())
            z_p, loss_np = z_p.detach().numpy(), loss.detach().numpy()
            graph = func.causal_graph(w_threshold=0.)
            
            fig, axs = plt.subplots(1,3, figsize=(10, 2.3))
            fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
            axs[0].plot(times[:100].squeeze(),data[:100].squeeze())
            axs[1].plot(z_p.squeeze())
            axs[1].set_title("Iteration = %i" % i + ",  " +  "Loss = %1.3f" % loss_np)
            cax = axs[2].matshow(graph)
            fig.colorbar(cax)
            plt.show()
            #plt.savefig('./Giff/fig%i.png' % i, bbox_inches = "tight")
    
            #utils.plot_trajectories(data[:100], z_p, graph, title=[i,loss_np])
            clear_output(wait=True)

            
def optimize(model, X, Y, lambda1, lambda2):
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y)
    def closure():
        optimizer.zero_grad()
        Y_hat = model(X_torch)
        loss = squared_loss(Y_hat, Y_torch)
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_reg()
        primal_obj = loss + l2_reg + l1_reg
        primal_obj.backward()
        return primal_obj
    optimizer.step(closure)