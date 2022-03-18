import numpy as np

def lingam_method(data):
    import lingam # https://github.com/cdt15/lingam/tree/master/lingam
    model = lingam.VARLiNGAM(lags=1, criterion='bic')
    model.fit(data)
    model.adjacency_matrices_[1][np.abs(model.adjacency_matrices_[1]) < 0.1] = 0
    return model.adjacency_matrices_[1]

