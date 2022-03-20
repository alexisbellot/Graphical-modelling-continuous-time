from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import UnivariateSpline

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True, penalty = "L1", GL_penalty = None): # dims: [number of variables, dimension hidden layers, output dim=1]
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.penalty = penalty
        self.GL_penalty = GL_penalty
        # fc1: variable splitting for l1
        self.fc1 = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        # x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i], i = number of hidden nodes
        # fc1_weight = self.fc1.weight
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_reg(self):
        """Take norm of fc1 weight"""
        if self.penalty == "L1":
            return torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        elif self.penalty == 'GL':
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
            # fc1_weight = self.fc1.weight
            self.fc1_recorded = torch.norm(fc1_weight, dim=1)
            return torch.sum(torch.norm(fc1_weight, dim=1)) # [j * m1]
        elif self.penalty == 'GL+AGL':
            gamma = 0.5
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
            # fc1_weight = self.fc1.weight
            return torch.sum(torch.norm(fc1_weight, dim=1) / (self.GL_penalty.pow(gamma))) # [j * m1]
    
    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        # fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
    
    def proximal(self, lam, eta=1e-2):
        """In place proximal update on first layer weight matrix"""
        fc1_weight = self.fc1.weight
        tmp = torch.norm(fc1.weight, dim=1) - lam*eta
        alpha = torch.clamp(tmp, min=0)
        v = torch.nn.functional.normalize(fc1_weight, dim=1)*alpha[:,None]
        self.fc1.weight.data = v



def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


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
    #proximal(lam=0.5)
    optimizer.step(closure)
    
def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      Y: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      w_threshold: float = 0.3):
    optimize(model,X,Y,lambda1,lambda2)
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return np.transpose(W_est)


def compute_derivatives(y,k=4,s=4,t=None):
    """Compute derivatives of univariate stochastic process by interpolating trajectory with
    univariate splines.

    Args:
        t, y (np.ndarray): time indeces t of time series y 
        
    Returns:
        dy/dt (np.ndarray): derivative of y(t) evaluated at t
    """
    if type(t) == type(None):
        t = np.arange(y.shape[0])
    
    temp_list = []
    for i in range(y.shape[1]):
        spl = UnivariateSpline(t, y[:,i], k=k, s=s)#s=0)
        derspl = spl.derivative()
        temp_list.append(derspl(t))
        
    return np.transpose(np.array(temp_list))


def nonlinear(data, s=0.01, lambda1=0.0001, lambda2=0.01, w_threshold = 0.15):
    
    # Rossler: lambda1=0.0001, lambda2=0.01, w_threshold = 0.15
    # Lorenz: lambda1=0.006, lambda2=0.01, w_threshold = 0.2
    Y = compute_derivatives(data,k=4,s=s)
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    d = data.shape[1]
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, data, Y, lambda1=lambda1, lambda2=lambda2, w_threshold = w_threshold)
    return W_est

