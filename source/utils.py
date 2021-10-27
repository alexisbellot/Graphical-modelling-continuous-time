import numpy as np
from scipy.special import expit as sigmoid
#import igraph as ig
import random
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import sdeint


def plot_trajectories(data, pred, graph, title=[1,2.1]):
    fig, axs = plt.subplots(1,3, figsize=(10, 2.3))
    fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
    axs[0].plot(data.squeeze())
    axs[1].plot(pred.squeeze())
    i=1
    axs[1].set_title("Iteration = %i" % title[0] + ",  " +  "Loss = %1.3f" % title[1])
    cax = axs[2].matshow(graph)
    fig.colorbar(cax)
    plt.show()
    #plt.savefig('../Giff/fig'+i+'.png')
    
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
        
    return np.transpose(np.array(temp_list)) # shape is number of time points x number of variables

def compute_spline(y,k=3,s=0.1):
    """Compute univariate stochastic process by interpolating trajectory with
    univariate splines.

    Args:
        t, y (np.ndarray): time indeces t of time series y 
        
    Returns:
        dy/dt (np.ndarray): derivative of y(t) evaluated at t
    """
    t = np.arange(y.shape[0])
    
    temp_list = []
    for i in range(y.shape[1]):
        spl = UnivariateSpline(t, y[:,i], k=k, s=s)
        temp_list.append(spl(t))
        
    return np.transpose(np.array(temp_list))

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC



def lorenz(x, t, F=5):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, sigma=0.5, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=None):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma]*p)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    #X = odeint(lorenz, x0, t, args=(F,))
    #X += np.random.normal(scale=sd, size=(T + burn_in, p))
    
    X = sdeint.itoint(lorenz, GG, x0, t)

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC

def lotkavolterra(x, t, r, alpha):
    '''Partial derivatives for Lotka-Volterra ODE.
    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = r[i]*x[i]*(1-np.dot(alpha[i],x))

    return dxdt

def simulate_lotkavolterra(p, T, r, alpha, delta_t=0.1, sd=0.01, burn_in=1000,
                       seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p) + 0.25
    x0 = np.array([0.0222, 0.0014, 0.0013, 0.0008])
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lotkavolterra, x0, t, args=(r,alpha,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = (alpha != 0)*1
    np.fill_diagonal(GC,1)

    return X[burn_in:], GC

def rossler(x, t, a=0,eps=0.1,b=4,d=2):
    '''Partial derivatives for rossler ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    dxdt[0] = a*x[0] - x[1]
    dxdt[p-2] = x[(p-3)]
    dxdt[p-1] = eps + b * x[(p-1)]*(x[(p-2)] - d)
    
    for i in range(1,p-2):
        dxdt[i] = np.sin(x[(i-1)]) - np.sin(x[(i+1)])
        
    return dxdt

def simulate_rossler(p, T, sigma=0.5, a=0, eps=0.1,b=4,d=2, delta_t=0.05, sd=0.1, burn_in=1000,
                       seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    def GG(x, t):
        p = len(x)
        return np.diag([sigma]*p)
    
    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    #X = odeint(rossler, x0, t, args=(a,eps,b,d,))
    #X += np.random.normal(scale=sd, size=(T + burn_in, p))

    X = sdeint.itoint(rossler, GG, x0, t)
    
    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    GC[0,0]=1; GC[0,1]=1
    GC[p-2,p-3]=1
    GC[p-1,p-1]=1; GC[p-1,p-2]=1
    for i in range(1,p-2):
        #GC[i, i] = 1
        GC[i, (i + 1)] = 1
        GC[i, (i - 1)] = 1
        
    return 400 * X[burn_in:], GC

def tumor_vaccine(x, t, c2, t1, a0=0.1946, a1=0.3, c1=100, c3=300, delta0= 0.00001, delta1= 0.00001, d=0.0007, f=0.62, r=0.01):
    '''Partial derivatives for rossler ODE.'''
        
    dxdt = np.zeros(5)
    
    c0=1/369
    dxdt[0] = a0*x[0]*(1 - c0*x[0]) - delta0*x[0]*x[2] / (1 + c1*x[1]) - delta0*x[0]*x[4]
    dxdt[1] = a1*(x[0]**2) / (c2 + x[0]**2) - d*x[1]
    dxdt[2] = f*x[2]*x[0] / (1 + c3*x[0]*x[1]) - r*x[2] - delta0*x[3]*x[2] - delta1*x[2]
    dxdt[3] = r*x[2] - delta1*x[3]
    
    if math.isclose(t, t1, abs_tol=0.5):
        dxdt[4] = 5000 - delta1*x[4]
    else:
        dxdt[4] = - delta1*x[4]
    
    
    
    return dxdt

def simulate_tumor(T, c2=300, t1=3,delta_t=0.05, sd=0.1, burn_in=0,
                       seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.zeros(5)
    x0[0] = 3; x0[1] = 0; x0[2] = 100; x0[3] = 0; x0[4] = 0
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(tumor_vaccine, x0, t, args=(c2, t1, ))
    X += np.random.normal(scale=sd, size=(T + burn_in, 5))

    # Set up Granger causality ground truth.
    p=5
    GC = np.zeros((p, p), dtype=int)
    GC[0,0]=1; GC[0,1]=1
    GC[p-2,p-3]=1
    GC[p-1,p-1]=1; GC[p-1,p-2]=1
    for i in range(1,p-2):
        #GC[i, i] = 1
        GC[i, (i + 1)] = 1
        GC[i, (i - 1)] = 1
        
    return X[burn_in:], GC

def glycolytic(x,t,k1=0.52, K1=100, K2=6, K3=16, K4=100, K5=1.28, K6=12, K=1.8, 
               kappa=13, phi=0.1, q=4, A=4, N=1, J0=2.5):
    '''Partial derivatives for Glycolytic oscillator model.
    
    source:
    https://www.pnas.org/content/pnas/suppl/2016/03/23/1517384113.DCSupplemental/pnas.1517384113.sapp.pdf
    
    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions'''
    dxdt = np.zeros(7)
    
    dxdt[0] = J0 - (K1*x[0]*x[5])/(1 + (x[5]/k1)**q)
    dxdt[1] = (2*K1*x[0]*x[5])/(1 + (x[5]/k1)**q) - K2*x[1]*(N-x[4]) - K6*x[1]*x[4]
    dxdt[2] = K2*x[1]*(N-x[4]) - K3*x[2]*(A-x[5])
    dxdt[3] = K3*x[2]*(A-x[5]) - K4*x[3]*x[4] - kappa*(x[3] - x[6])
    dxdt[4] = K2*x[1]*(N-x[4]) - K4*x[3]*x[4] - K6*x[1]*x[4]
    dxdt[5] = (-2*K1*x[0]*x[5])/(1 + (x[5]/k1)**q) + 2*K3*x[2]*(A-x[5]) - K5*x[5]
    dxdt[6] = phi*kappa*(x[3]-x[6]) - K*x[6]

    return dxdt

def simulate_glycolytic(T, sigma = 0.5, delta_t=0.001, sd=0.01, burn_in=0,seed=None, scale=True):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma]*p)
    
    x0 = np.zeros(7)
    x0[0] = np.random.uniform(0.15, 1.6)
    x0[1] = np.random.uniform(0.19, 2.16)
    x0[2] = np.random.uniform(0.04, 0.2)
    x0[3] = np.random.uniform(0.1, 0.35)
    x0[4] = np.random.uniform(0.08, 0.3)
    x0[5] = np.random.uniform(0.14, 2.67)
    x0[6] = np.random.uniform(0.05, 0.1)    
    
    # Use scipy to solve ODE.
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    #X = odeint(glycolytic, x0, t)
    #X += np.random.normal(scale=sd, size=(T + burn_in, 7))
    
    X = sdeint.itoint(glycolytic, GG, x0, t)
    
    # Set up ground truth.
    GC = np.zeros((7, 7), dtype=int)
    GC[0,:] = np.array([1,0,0,0,0,1,0])
    GC[1,:] = np.array([1,1,0,0,1,1,0])
    GC[2,:] = np.array([0,1,1,0,1,1,0])
    GC[3,:] = np.array([0,0,1,1,1,1,1])
    GC[4,:] = np.array([0,1,0,0,1,1,0])
    GC[5,:] = np.array([1,1,0,0,0,1,0])
    GC[6,:] = np.array([0,0,0,1,0,0,1])
    
    if scale:
        X = np.transpose(np.array([(X[:,i] - X[:,i].min())/(X[:,i].max() - X[:,i].min()) for i in range(X.shape[1])]))
   
    return 10* X[burn_in:], GC
                        
def cardiovascular(x,t,I_ext, Rmod, Ca=4, Cv=111, tau=20, k=0.1838, Pas=70):
    '''Partial derivatives for Glycolytic oscillator model.
    
    source:
    https://www.pnas.org/content/pnas/suppl/2016/03/23/1517384113.DCSupplemental/pnas.1517384113.sapp.pdf
    
    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions'''
    dxdt = np.zeros(4)
    
    def f(S, maxx=3, minn=0.66):
        return S*(maxx-minn) + minn
    
    def R(S, Rmod, Rmax=2.134, Rmin=0.5335):
        return S*(Rmax - Rmin) + Rmin + Rmod
    
    dxdt[0] = I_ext
    dxdt[1] = (1/Ca)*((x[1] - x[2])/R(x[3], Rmod) - x[0]*f(x[3]))
    dxdt[2] = (1/Cv)*(-Ca*dxdt[1] + I_ext)
    dxdt[3] = (1/tau)*(1-1/(1 + np.exp(-k*(x[1] - Pas))) - x[3])

    return dxdt
    

def simulate_cardiovascular(T, delta_t=0.001, sd=0.01, burn_in=0,seed=None, scale=True):
    if seed is not None:
        np.random.seed(seed)

    x0 = np.zeros(4)
    x0[0] = np.random.uniform(90,100)
    x0[1] = np.random.uniform(75,85)
    x0[2] = np.random.uniform(3,7)
    x0[3] = np.random.uniform(0.15,0.25)   
    
    I_ext, Rmod = np.random.choice([-2,0]), np.random.choice([0.5,0])
    
    # Use scipy to solve ODE.
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(cardiovascular, x0, t, args=(I_ext, Rmod,))
    X += np.random.normal(scale=sd, size=(T + burn_in, 4))

    # Set up ground truth.
    GC = np.zeros((4, 4), dtype=int)
    GC[0,:] = np.array([0,0,0,0])
    GC[1,:] = np.array([1,1,1,1])
    GC[2,:] = np.array([0,1,0,0])
    GC[3,:] = np.array([0,0,1,1,1,1,1])
    
    if scale:
        X = np.transpose(np.array([(X[:,i] - X[:,i].min())/(X[:,i].max() - X[:,i].min()) for i in range(X.shape[1])]))
   
    return 10* X[burn_in:], GC

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)




def compare_graphs(true_graph, estimated_graph):
    '''Compute performance measures on (binary) adjacency matrix
    Input:
     - true_graph: (dxd) np.array, the true adjacency matrix
     - estimated graph: (dxd) np.array, the estimated adjacency matrix (weighted or unweighted)
    '''
    
    def structural_hamming_distance(W_true, W_est):
        '''Computes the structural hamming distance'''

        pred = np.flatnonzero(W_est != 0)
        cond = np.flatnonzero(W_true)
        cond_reversed = np.flatnonzero(W_true.T)

        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

        pred_lower = np.flatnonzero(np.tril(W_est + W_est.T))
        cond_lower = np.flatnonzero(np.tril(W_true + W_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        return shd
    
    num_edges = len(true_graph[np.where(true_graph != 0.0)])

    tam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in true_graph])
    eam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in estimated_graph])

    tp = len(np.argwhere((tam + eam) == 2))
    fp = len(np.argwhere((tam - eam) < 0))
    tn = len(np.argwhere((tam + eam) == 0))
    fn = num_edges - tp
    x = [tp, fp, tn, fn]
    
    if x[0] + x[1] == 0: precision = 0 
    else: precision = float(x[0]) / float(x[0] + x[1])
    if tp + fn == 0: tpr = 0
    else: tpr = float(tp) / float(tp + fn) 
    if x[2] + x[1] == 0: specificity = 0
    else: specificity = float(x[2]) / float(x[2] + x[1])
    if precision + tpr == 0: f1 = 0
    else: f1 = 2 * precision * tpr / (precision + tpr)
    if fp + tp == 0: fdr = 0
    else: fdr = float(fp)/(float(fp) + float(tp))
        
    shd = structural_hamming_distance(true_graph, estimated_graph)
    
    AUC = roc_auc_score(true_graph.flatten(),estimated_graph.flatten())
    
    return tpr, fdr, AUC


def performance(methods, model='rossler', num_exp=100, T = 100, p = 10, F=50):
    ''' Performance comparisons on estimation of W with different methods'''
    
    performance = defaultdict(float)
    
    for exp in tqdm(range(num_exp)):
        
        if model == 'rossler':
            data, GC = simulate_rossler(p=p, a=0, T=T, delta_t=0.05, sd=0.00, burn_in=0)
            lambda1=0.0001
        if model == 'lorenz':
            data, GC = simulate_lorenz(p=p, T=T, F=F)
            lambda1=0.006
        if model == 'glycolytic':
            data, GC = simulate_glycolytic(T=T)
            lambda1=0.0001
        
        for method in methods:
            method_name = method.__name__
            key = 'method: {}; '.format(method_name)
            tic = time.time()
            if method == 'nonlinear':
                w_est = method(data, lambda1=lambda1)
            else:
                w_est = method(data)
            toc = (time.time() - tic)
            
            shd = compare_graphs(GC, w_est)
            performance[key] += shd / num_exp
            
    return performance

