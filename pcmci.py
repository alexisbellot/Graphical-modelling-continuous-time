import numpy as np

import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI


def pcmci(data):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
    q_matrix = (q_matrix < 0.001) * 1  # add 00
    return np.transpose(q_matrix[:, :, 2])
