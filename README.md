# Graphical modelling in continuous-time

This is a python implementation of algorithms and experiments presented in the paper [*"Graphical modelling in continuous-time: consistency guarantees and algorithms using Neural ODEs"*](https://arxiv.org/abs/2105.02522). 

Graphical modelling is the problem of defining and piecing together associations in data to infer the underlying structure among a system of variables. This project considers score-based graph learning for the study of dynamical systems. The proposal is a score-based learning algorithm based on penalized Neural Ordinary Differential Equations that we show to be applicable to the general setting of irregularly-sampled multivariate time series.


## Installation
This project uses `pytorch` and `torchdiffeq`. For full list of dependencies see [`requirements.txt`](./requirements.txt) or [`environment.yml`](./environment.yml) (for `conda`). In order to run the model and the paper experiments, install the dependencies from the appropriate file.

## First steps
To get started, check [`Tutorial.ipynb`](./experiments/Tutorial.ipynb) which will guide you through graphical modelling in continuous-time from the beginning.

For the experiments, see [`Paper_experiments_Lorenz.ipynb`](./experiments/Paper_experiments_Lorenz.ipynb) and [`Paper_experiments_Rossler.ipynb`](./experiments/Paper_experiments_Rossler.ipynb).
